import serial
import time
from collections import deque
import numpy as np
import os

# ====== 你要改的設定 ======
COM_PORT = "COM9"        # <<< 改成你在裝置管理員看到的 BrainLink COM port
BAUD_RATE = 57600        # BrainLink Lite / TGAM 預設 57600
LATEST_SEGMENT_PATH = "latest_eeg.npy"

# 你現在用 500 Hz（跟 SVM / EEGNet 設定一致）
SAMPLE_RATE = 500

# 一個樣本的長度（秒） → non-overlap 的窗長
WINDOW_SECONDS = 2.0
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)

# NEW: 兩個樣本之間「額外跳過」的時間（秒）
#   - 設 0.0  = 完全貼齊：4 秒 → 4 秒 → 4 秒（不重疊）
#   - 設 2.0  = 4 秒樣本 + 空 2 秒 → 下一個 4 秒樣本（決策每 6 秒一次）
NON_OVERLAP_GAP_SECONDS = 0.0
GAP_SAMPLES = int(SAMPLE_RATE * NON_OVERLAP_GAP_SECONDS)

# === NEW: Blinker 相關設定 ===
BLINK_DETECT_WINDOW_SECONDS = 2.0
BLINK_DETECT_WINDOW_SIZE = int(SAMPLE_RATE * BLINK_DETECT_WINDOW_SECONDS)
BLINK_REFRACTORY_SECONDS = 2.0              # 兩次 blink 之間至少隔這麼久才再觸發
BLINK_TIMESTAMP_PATH = "latest_blink.txt"   # 寫入最近一次眨眼的時間戳（秒）


class ThinkGearParser:
    """
    純 Python 版 ThinkGear Packet Parser
    參考 NeuroSky ThinkGear Communications Protocol 文件
    會從 serial byte stream 中抓出 0x80 (RAW wave) 的 sample
    """

    def __init__(self, on_raw=None, on_attention=None, on_meditation=None):
        self.buffer = bytearray()
        self.on_raw = on_raw
        self.on_attention = on_attention
        self.on_meditation = on_meditation

    def feed(self, data: bytes):
        """丟進一段新讀到的 bytes，就會自動解析裡面全部的封包"""
        self.buffer.extend(data)

        while True:
            # 至少要看到 header [0xAA, 0xAA, PLEN]
            if len(self.buffer) < 4:
                return

            # 尋找第一個 0xAA
            try:
                idx = self.buffer.index(0xAA)
            except ValueError:
                # 找不到 0xAA，整個 buffer 丟掉
                self.buffer.clear()
                return

            if idx > 0:
                # 前面都是雜訊，丟掉
                del self.buffer[:idx]

            if len(self.buffer) < 3:
                # 還不夠 header
                return

            # 確認第二個 SYNC
            if self.buffer[0] != 0xAA or self.buffer[1] != 0xAA:
                # 理論上不會到這，但保險一點
                del self.buffer[0]
                continue

            payload_len = self.buffer[2]
            if payload_len > 169:
                # 不合理，可能 mis-sync，往後丟一個 byte 再找
                del self.buffer[0]
                continue

            packet_len = 3 + payload_len + 1  # header(3) + payload + checksum
            if len(self.buffer) < packet_len:
                # 封包還沒收完
                return

            payload = self.buffer[3: 3 + payload_len]
            checksum = self.buffer[3 + payload_len]

            # 驗證 checksum
            s = sum(payload) & 0xFF
            calc_checksum = (~s) & 0xFF
            if calc_checksum != checksum:
                # checksum 錯，捲掉一個 byte 重找 sync
                del self.buffer[0]
                continue

            # 到這裡代表這個封包是合法的，解析 payload
            self._parse_payload(payload)

            # 把這個封包吃掉，處理下一個
            del self.buffer[:packet_len]

    def _parse_payload(self, payload: bytes):
        i = 0
        n = len(payload)

        while i < n:
            # EXCODE (0x55) 先略過
            while i < n and payload[i] == 0x55:
                i += 1  # 目前不需要 excode 等級

            if i >= n:
                break

            code = payload[i]
            i += 1

            # code >= 0x80 → 有 vlen
            if code >= 0x80:
                if i >= n:
                    break
                vlen = payload[i]
                i += 1
                if i + vlen > n:
                    break
                value = payload[i: i + vlen]
                i += vlen
            else:
                # 單一 byte value
                if i >= n:
                    break
                vlen = 1
                value = payload[i: i + 1]
                i += 1

            # 根據 code 做對應處理
            if code == 0x80 and vlen == 2:
                # RAW wave: 16-bit signed big-endian
                high, low = value[0], value[1]
                raw = high * 256 + low
                if raw >= 32768:
                    raw -= 65536
                if self.on_raw is not None:
                    self.on_raw(raw)

            elif code == 0x04 and vlen == 1:
                # Attention (0~100)
                att = value[0]
                if self.on_attention is not None:
                    self.on_attention(att)

            elif code == 0x05 and vlen == 1:
                # Meditation (0~100)
                med = value[0]
                if self.on_meditation is not None:
                    self.on_meditation(med)

            # 其他 code 先不處理（0x02 poor signal, 0x83 EEG power 等）


# === NEW: Blinker 演算法（跟 v6 版邏輯一致，只用 numpy） ===
def detect_blinks_blinker(
    segment,
    fs=SAMPLE_RATE,
    threshold_mad=7.0,
    min_duration=0.06,
    max_duration=0.40,
):
    """
    簡易眨眼偵測，回傳在這個 segment 中偵測到的眨眼數量。
    """
    x = np.asarray(segment, dtype=float)
    if x.ndim != 1:
        x = np.ravel(x)
    if x.size < int(min_duration * fs):
        return 0

    x = x - np.median(x)
    mad = np.median(np.abs(x)) + 1e-6
    x_norm = x / mad

    win = max(1, int(0.02 * fs))
    if win > 1:
        kernel = np.ones(win, dtype=float) / win
        x_smooth = np.convolve(x_norm, kernel, mode="same")
    else:
        x_smooth = x_norm

    above = np.where(np.abs(x_smooth) > threshold_mad)[0]
    if above.size == 0:
        return 0

    blink_count = 0
    start = above[0]
    for i in range(1, len(above)):
        if above[i] != above[i - 1] + 1:
            dur = (above[i - 1] - start + 1) / fs
            if min_duration <= dur <= max_duration:
                blink_count += 1
            start = above[i]
    dur = (above[-1] - start + 1) / fs
    if min_duration <= dur <= max_duration:
        blink_count += 1

    return blink_count


def main():
    # buffer_seg：專門拿來切 non-overlap segment 給 EEGNet 用
    buffer_seg = deque()
    # buffer_blink：只保留最近 BLINK_DETECT_WINDOW_SECONDS 秒，用來做眨眼偵測
    buffer_blink = deque(maxlen=BLINK_DETECT_WINDOW_SIZE)

    # NEW: 最近一次眨眼時間
    last_blink_time = 0.0

    def handle_raw(sample: int):
        nonlocal last_blink_time

        # ---------- 1) non-overlap segment 給 EEGNet ----------
        buffer_seg.append(sample)

        # 需要的總長度 = WINDOW_SIZE (樣本長度) + GAP_SAMPLES (間隔要丟掉的樣本)
        if len(buffer_seg) >= WINDOW_SIZE + GAP_SAMPLES:
            # 1) 取出前 WINDOW_SIZE 個 sample 當成一個完整 segment
            seg_list = [buffer_seg.popleft() for _ in range(WINDOW_SIZE)]
            seg = np.array(seg_list, dtype=np.float32)
            seg = seg - np.mean(seg)

            tmp_path = LATEST_SEGMENT_PATH + ".tmp"
            try:
                with open(tmp_path, "wb") as f:
                    np.save(f, seg)
                os.replace(tmp_path, LATEST_SEGMENT_PATH)
                # print(f"[INFO] saved non-overlap segment ({WINDOW_SECONDS}s) to {LATEST_SEGMENT_PATH}")
            except PermissionError:
                # 如果剛好被讀檔鎖住，就略過這次寫檔
                pass
            except Exception as e:
                print("[ERROR] 寫入 latest_eeg.npy 時發生其他錯誤：", e)

            # 2) 丟掉 GAP_SAMPLES（間隔） → 控制樣本之間的距離
            for _ in range(GAP_SAMPLES):
                if not buffer_seg:
                    break
                buffer_seg.popleft()

        # ---------- 2) Blinker：在最近 BLINK_DETECT_WINDOW_SECONDS 秒的資料上偵測眨眼 ----------
        buffer_blink.append(sample)

        if len(buffer_blink) >= BLINK_DETECT_WINDOW_SIZE:
            seg2 = np.array(buffer_blink, dtype=np.float32)
            n_blinks = detect_blinks_blinker(seg2, fs=SAMPLE_RATE)

            if n_blinks >= 1:
                now = time.time()
                if (now - last_blink_time) >= BLINK_REFRACTORY_SECONDS:
                    last_blink_time = now
                    try:
                        with open(BLINK_TIMESTAMP_PATH, "w") as f:
                            f.write(str(now))
                        # print(f"[INFO] Blink detected ({n_blinks}), timestamp updated.")
                    except Exception as e:
                        print("[WARN] 寫入 blink timestamp 失敗：", e)

    # 若你之後想在終端顯示 Attention / Meditation 可以在這裡加
    def handle_attention(att):
        pass
        # print("Attention:", att)

    def handle_meditation(med):
        pass
        # print("Meditation:", med)

    parser = ThinkGearParser(
        on_raw=handle_raw,
        on_attention=handle_attention,
        on_meditation=handle_meditation,
    )

    print(f"[INFO] Opening {COM_PORT} @ {BAUD_RATE} ...")
    with serial.Serial(COM_PORT, BAUD_RATE, timeout=1) as ser:
        print("[INFO] Connected. Start reading BrainLink data...")
        while True:
            data = ser.read(1024)
            if not data:
                continue
            parser.feed(data)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
