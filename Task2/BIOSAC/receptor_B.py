# server_B_simple_low_latency.py
import socket
import numpy as np
from BCI_SVM_REV_selection_aug_proba_wavelet import Config  # 取得 SAMPLING_RATE / SEGMENT_LENGTH
import datetime
from pathlib import Path

LATEST_SEGMENT_PATH = Path("latest_eeg.npy")
SEGMENT_SAMPLES = int(Config.SEGMENT_LENGTH * Config.SAMPLING_RATE)

# 全域 buffer，用來存最近一段 EEG
eeg_buffer = np.array([], dtype=np.float32)

HOST = "0.0.0.0"
PORT = 50007
SAVE_TO_FILE = True
OUTPUT_FILE = Path("eeg_received_log.txt")

def parse_eeg_from_text(raw_text: str) -> np.ndarray:
    """
    把收到的 EEG 文字（可能有空白、tab、逗號）轉成 1D float array。
    解析不了的 token 就略過。
    """
    tokens = raw_text.replace(",", " ").split()
    vals = []
    for t in tokens:
        try:
            vals.append(float(t))
        except ValueError:
            continue
    return np.array(vals, dtype=np.float32)


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # === 低延遲相關設定（新增） ===
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)        # 方便重啟 server
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)        # 關掉 Nagle，減少封包延遲
        # 可選：調整 buffer，大多數情況可省略或用預設
        # s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024)  # 8 KB
        # s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024)  # 8 KB
        # ==========================

        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[Server] Listening on {HOST}:{PORT} ...")

        conn, addr = s.accept()
        with conn:
            print(f"[Server] Connected by {addr}")

            # 對每個連進來的 conn 再保險設一次 TCP_NODELAY（習慣用法）
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            while True:
                data = conn.recv(1024 * 1024)
                if not data:
                    print("[Server] Connection closed by client.")
                    break

                raw_text = data.decode("utf-8", errors="replace")
                if not raw_text.strip():
                    continue

                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n===== [{ts}] Received EEG Text =====")
                preview = raw_text[:500]
                print(preview + ("..." if len(raw_text) > 500 else ""))
                print("====================================\n")

                                # 1. 解析 EEG 數值
                new_samples = parse_eeg_from_text(raw_text)
                if new_samples.size > 0:
                    global eeg_buffer
                    # 接到的新樣本接在後面
                    eeg_buffer = np.concatenate([eeg_buffer, new_samples])

                    # 為了避免爆記憶體，只保留最近 3 個 window 的長度
                    max_buf = SEGMENT_SAMPLES * 3
                    if eeg_buffer.size > max_buf:
                        eeg_buffer = eeg_buffer[-max_buf:]

                    # 2. 當 buffer 長度夠長，就取最後一段寫成 latest_eeg.npy
                    if eeg_buffer.size >= SEGMENT_SAMPLES:
                        latest_seg = eeg_buffer[-SEGMENT_SAMPLES:]
                        np.save(LATEST_SEGMENT_PATH, latest_seg)
                        # print("saved latest_eeg.npy", latest_seg.shape)
                
                if SAVE_TO_FILE:
                    with OUTPUT_FILE.open("a", encoding="utf-8") as f:
                        f.write(f"\n===== [{ts}] =====\n")
                        f.write(raw_text)
                        f.write("\n")


if __name__ == "__main__":
    main()
