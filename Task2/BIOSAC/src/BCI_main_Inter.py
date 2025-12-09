import threading
import queue
import time
import tkinter as tk
from tkinter import ttk
import os
import sys
import numpy as np
from joblib import load

from BCI_SVM_REV_selection_aug_proba_wavelet import (
    Config as SVMConfig,
    preprocess_segment,
    extract_features,
    BandFeatureSVMClassifier,
)
import serial  # pip install pyserial

sys.modules["__main__"].BandFeatureSVMClassifier = BandFeatureSVMClassifier

# ====== 預設狀態：不更動，用來讓你用鍵盤測 GUI ======
DEFAULT_STATE = "idle"
SVM_MODEL = None
EEG_SEG_LEN = int(SVMConfig.SEGMENT_LENGTH * SVMConfig.SAMPLING_RATE)

LATEST_SEGMENT_PATH = "latest_eeg.npy"  # receptor 寫的檔名
LAST_PRED_STATE = DEFAULT_STATE
LAST_BLINK_TS = 0.0


# ====== 初始化已訓練好的模型 ========
def init_svm_model(model_path="bci_svm.pkl"):
    """
    程式啟動時呼叫一次，載入訓練好的 SVM 模型。
    """
    global SVM_MODEL

    if not os.path.exists(model_path):
        print(f"[WARN] 找不到 SVM 模型檔 {model_path}，改用鍵盤控制。")
        SVM_MODEL = None
        return

    SVM_MODEL = load(model_path)
    print(f"[INFO] SVM 模型載入完成：{model_path}")


def classifier_predict(_unused=None):
    """
    從 latest_eeg.npy 讀取一段 EEG，
    經過與離線訓練相同的 preprocess + feature 抽取，
    丟進訓練好的 SVM，輸出 'relax' / 'focus'。
    """
    global LAST_PRED_STATE, LAST_BLINK_TS

    # 沒有載入模型就直接用預設狀態（只用鍵盤）
    if SVM_MODEL is None:
        return DEFAULT_STATE

    # 沒有新檔就維持上一次的 state
    if not os.path.exists(LATEST_SEGMENT_PATH):
        return LAST_PRED_STATE

    try:
        seg = np.load(LATEST_SEGMENT_PATH)  # [T]
    except Exception as e:
        print("[WARN] 讀取最新 EEG 檔失敗：", e)
        return LAST_PRED_STATE

    # 確保是一維、長度夠
    if seg.ndim != 1 or len(seg) < EEG_SEG_LEN:
        return LAST_PRED_STATE

    # 比設定長就取最後 EEG_SEG_LEN 個 sample
    if len(seg) > EEG_SEG_LEN:
        seg = seg[-EEG_SEG_LEN:]

    has_blink, blink_count = detect_blinks_blinker_like(
        seg,
        fs=SVMConfig.SAMPLING_RATE,
        window_sec = 2.0,
        min_blinks_in_window = 3
    )
    
    now_ts = time.time()
    BLINK_REFRACTORY = 1.2
    
    if has_blink and (now_ts - LAST_BLINK_TS) > BLINK_REFRACTORY: 
        LAST_BLINK_TS = now_ts
        print(f"[BLINKER] Detected blink events: count={blink_count} -> trigger Enter")
        return "blink"
      
    # 跟離線訓練完全一樣的 preprocess
    seg_proc = preprocess_segment(seg, fs=SVMConfig.SAMPLING_RATE)

    # 抽 feature（35 維）→ [1, 35]
    feats = extract_features(seg_proc, fs=SVMConfig.SAMPLING_RATE).reshape(1, -1)

    # 丟進 SVM 做預測
    # 你也可以改用 decision_function 或 predict_proba 做信心門檻
    pred_class = int(SVM_MODEL.predict(feats)[0])

    if pred_class == 0:
        state = "relax"
    elif pred_class == 1:
        state = "focus"
    else:
        state = DEFAULT_STATE

    LAST_PRED_STATE = state
    return state

def detect_blinks_blinker_like(
    seg,
    fs=SVMConfig.SAMPLING_RATE,
    window_sec: float = 2.0,
    min_blink_duration: float = 0.04,
    max_blink_duration: float = 0.4,
    min_inter_blink_interval: float = 0.2,
    threshold_k: float = 2.5,
    min_blinks_in_window: int = 3,
):
    """
    簡化版 Blinker 演算法：
    - 只看最近 window_sec 秒的資料
    - 以幅度門檻 + 持續時間找出「眨眼樣的波形」
    - 合併過於接近的事件，得到眨眼次數

    預設條件：兩秒內偵測到 >= min_blinks_in_window 次 blink 才視為有效指令。
    可以依實驗狀況調整 min_blinks_in_window 或 threshold_k 來改靈敏度。
    """
    seg = np.asarray(seg, dtype=float)
    if seg.size < int(fs * 0.5):
        # 資料太短，不判斷
        return False, 0

    # 只保留最後 window_sec 秒
    win_len = int(window_sec * fs)
    if seg.size > win_len:
        seg = seg[-win_len:]

    # 去掉直流，使用 median 比較 robust
    x = seg - np.median(seg)
    absx = np.abs(x)

    # 以 median + k * std 作為門檻
    eps = 1e-12
    sigma = np.std(absx)
    if sigma < eps:
        return False, 0
    thr = np.median(absx) + threshold_k * sigma

    binary = absx > thr
    if not binary.any():
        return False, 0

    # 找出連續 True run -> 候選 blink 區段
    idx = np.where(binary)[0]
    starts = []
    ends = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            starts.append(start)
            ends.append(prev)
            start = i
            prev = i
    starts.append(start)
    ends.append(prev)

    # 依長度篩選：典型 blink 約 40~400 ms
    candidates = []
    for s, e in zip(starts, ends):
        dur = (e - s + 1) / fs
        if min_blink_duration <= dur <= max_blink_duration:
            candidates.append((s, e))

    if not candidates:
        return False, 0

    # 合併間隔太短的候選（避免一次眨眼被切成多段）
    merged = []
    cur_s, cur_e = candidates[0]
    for s, e in candidates[1:]:
        gap = (s - cur_e) / fs
        if gap < min_inter_blink_interval:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    blink_count = len(merged)
    has_blink = blink_count >= min_blinks_in_window
    return has_blink, blink_count


class ArduinoLightController:
    """
    Serial 通訊協定:
        PWR:<0/1>       # power off/on
        RGB:<r>,<g>,<b> # base color (0-255)
        BRT:<0-255>     # global brightness
    """

    def __init__(self, port="COM5", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None

    def connect(self):
        if self.ser is None or not self.ser.is_open:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2.0)  # 等 Arduino reset

    def close(self):
        if self.ser is not None and self.ser.is_open:
            self.ser.close()

    def _send_line(self, line: str):
        if self.ser is None or not self.ser.is_open:
            print("[WARN] Arduino not connected, skip:", line.strip())
            return
        if not line.endswith("\n"):
            line += "\n"
        self.ser.write(line.encode("utf-8"))

    def set_power(self, on: bool):
        val = 1 if on else 0
        self._send_line(f"PWR:{val}")

    def set_brightness(self, level: int):
        level = max(0, min(255, int(level)))
        self._send_line(f"BRT:{level}")

    def set_color(self, r: int, g: int, b: int):
        r = max(0, min(255, int(r)))
        g = max(0, min(255, int(g)))
        b = max(0, min(255, int(b)))
        self._send_line(f"RGB:{r},{g},{b}")


class EEGStateSource(threading.Thread):
    """
    背景 thread：一直叫 classifier_predict，丟 state 到 queue 給 GUI 用。
    """

    def __init__(self, state_queue, stop_event):
        super().__init__(daemon=True)
        self.state_queue = state_queue
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            state = classifier_predict(None)
            self.state_queue.put(state)
            time.sleep(1.0)


class BCILightGUI:
    """
    三個主頁：
        0: 整體亮度
        1: 色溫
        2: 狀態調整

    狀態：
        模式: 固定 / 熄燈 / 調整
        整體亮度: 25%, 50%, 75%, 100% -> global brightness
        色溫: R / G / B 各自 25%, 50%, 75%, 100% -> 以 255 為 100%
    """

    MAIN_PAGES = ["整體亮度", "色溫", "狀態調整"]
    BRIGHTNESS_LEVELS = [0.25, 0.5, 0.75, 1.0]
    BRIGHTNESS_LABELS = ["25%", "50%", "75%", "100%"]
    COLOR_CHANNELS = ["R", "G", "B"]
    COLOR_PERCENT_LEVELS = [0.25, 0.5, 0.75, 1.0]
    COLOR_PERCENT_LABELS = ["25%", "50%", "75%", "100%"]
    MODES = ["固定", "熄燈", "調整"]  # 0,1,2

    # 顏色設定
    COLOR_PAGE_NORMAL = "#e0e0e0"
    COLOR_PAGE_ACTIVE = "#cfe2ff"      # 目前主頁
    COLOR_PAGE_CANDIDATE = "#ffe699"   # 主頁調整模式中的預選頁

    COLOR_ITEM_NORMAL = "#f7f7f7"
    COLOR_ITEM_SELECTED = "#cfe2ff"    # 游標所在子項目
    COLOR_ITEM_ADJUST = "#ffe699"      # 正在調整的子項目

    def __init__(self, root, arduino_port="COM6"):
        self.root = root
        self.arduino = ArduinoLightController(port=arduino_port)

        # classifier thread & queue
        self.state_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.classifier_thread = EEGStateSource(self.state_queue, self.stop_event)

        # --- 主頁 index：0=整體亮度,1=色溫,2=狀態調整 ---
        self.current_main_index = 2   # 預設在「狀態調整」

        # --- 每一頁的「選取子項目 index」---
        # 整體亮度頁：0=整體亮度, 1=調整項目
        self.sel_brightness = 0
        # 色溫頁：0=R,1=G,2=B,3=調整項目
        self.sel_color = 0
        # 狀態頁：0=固定,1=熄燈,2=調整,3=調整項目
        self.sel_status = 1  # 一開始指向熄燈

        # --- 調整模式 ---
        self.in_adjust_mode = False        # True: 正在調整
        self.adjust_context = None         # 'page' / 'brightness' / 'color' / None

        # 主頁候選 index（主頁切換模式用）
        self.page_cursor_index = self.current_main_index

        # --- 狀態調整頁的實際模式 ---
        self.mode_index = 1  # 初始為「熄燈」

        # --- 整體亮度頁（已確定的值 + 調整模式下候選值） ---
        self.brightness_index = 3          # 0~3, 真正生效的亮度檔位
        self.brightness_cursor_index = 3   # 調整模式下的候選亮度

        # --- 色溫頁 ---
        self.selected_channel_index = 0                  # 最近一次調的通道
        self.channel_percent_indices = [3, 3, 3]         # 每個通道已確定的 25/50/75/100 index
        self.color_percent_cursor_index = 3              # 調整模式下，某通道的候選比例

        # 建 GUI
        self._build_widgets()

        # 鍵盤 debug：↑ / ↓ / Enter
        self.root.bind("<Up>", lambda e: self.handle_up())
        self.root.bind("<Down>", lambda e: self.handle_down())
        self.root.bind("<Return>", lambda e: self.handle_enter())

        # 連 Arduino
        try:
            self.arduino.connect()
        except Exception as e:
            print("[WARN] Cannot open Arduino:", e)

        # 啟動 classifier thread
        self.classifier_thread.start()
        self.root.after(100, self._poll_state_queue)

        # 套用初始狀態（熄燈）
        self._on_mode_changed()
        self._refresh_ui()

    # ---------- GUI ----------
    def _build_widgets(self):
        self.root.title("BCI WS2812B Controller")

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # classifier 狀態顯示
        self.state_label_var = tk.StringVar(value="Classifier state: (waiting)")
        ttk.Label(main_frame, textvariable=self.state_label_var, font=("Segoe UI", 12)).grid(
            row=0, column=0, sticky="w"
        )

        # 自訂主功能列 tab（像 Word 上方功能列），用 Label 排成一排，用顏色做狀態
        self.tabs_frame = ttk.Frame(main_frame)
        self.tabs_frame.grid(row=1, column=0, sticky="ew", pady=(5, 5))
        self.tab_labels = []
        for i, name in enumerate(self.MAIN_PAGES):
            lbl = tk.Label(
                self.tabs_frame,
                text=name,
                bd=1,
                relief="raised",
                padx=10,
                pady=3
            )
            lbl.grid(row=0, column=i, padx=2)
            # 也支援滑鼠點擊切換主頁（只當作切換 current_main_index + reset 調整模式）
            lbl.bind("<Button-1>", lambda e, idx=i: self._on_tab_clicked(idx))
            self.tab_labels.append(lbl)

        # 子項目區：最多顯示 4 個項目（根據不同頁面內容更新文字與顏色）
        self.items_frame = ttk.Frame(main_frame)
        self.items_frame.grid(row=2, column=0, sticky="nsew", pady=(5, 5))
        self.item_labels = []
        for i in range(4):
            lbl = tk.Label(
                self.items_frame,
                text="",
                anchor="w",
                padx=8,
                pady=3,
                bd=1,
                relief="solid"
            )
            lbl.grid(row=i, column=0, sticky="ew", pady=2)
            self.item_labels.append(lbl)

        # 狀態 info
        self.info_label_var = tk.StringVar()
        ttk.Label(main_frame, textvariable=self.info_label_var, justify="left").grid(
            row=3, column=0, sticky="w", pady=(5, 5)
        )

        # 操作說明（簡化文字，主要靠顏色顯示）
        ttk.Label(
            main_frame,
            text=("控制方式 (BCI 對應)：Relax = 上, Concentrate = 下\n"
                  "主頁切換：在各頁的「調整項目」按 Enter 進入主頁調整模式，"
                  "此時上方主功能列會以不同底色顯示預選頁面，再按 Enter 確認。"),
            justify="left",
        ).grid(row=4, column=0, sticky="w", pady=(5, 0))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _on_tab_clicked(self, idx: int):
        # 滑鼠點 tab = 直接切換主頁，不走 BCI 流程
        if 0 <= idx < len(self.MAIN_PAGES):
            self.current_main_index = idx
            self.page_cursor_index = idx
            self.in_adjust_mode = False
            self.adjust_context = None
            self._refresh_ui()

    # ---------- BCI ----------
    def _poll_state_queue(self):
        try:
            while True:
                state = self.state_queue.get_nowait()
                self._handle_classifier_state(state)
        except queue.Empty:
            pass

        if not self.stop_event.is_set():
            self.root.after(100, self._poll_state_queue)

    def _handle_classifier_state(self, state: str):
        self.state_label_var.set(f"Classifier state: {state}")
        s = state.lower().strip()

        # 預設 idle：不動作
        if s in ("idle", "none", "no_change", ""):
            return

        if s == "relax":
            self.handle_up()
        elif s in ("concentrate", "focus"):
            self.handle_down()
        elif s in ("blink", "eye-blink", "eye_blink"):
            self.handle_enter()
        else:
            return

    # ---------- 導航入口 ----------
    def handle_up(self):
        if not self.in_adjust_mode:
            self._browse_up()
        else:
            self._adjust_up()
        self._refresh_ui()

    def handle_down(self):
        if not self.in_adjust_mode:
            self._browse_down()
        else:
            self._adjust_down()
        self._refresh_ui()

    def handle_enter(self):
        if not self.in_adjust_mode:
            # ===== 瀏覽模式：依目前頁面 + 選取項目 決定進哪種調整 / 動作 =====
            page = self.current_main_index
            if page == 0:  # 整體亮度
                if self.sel_brightness == 0:
                    # 選在「整體亮度」 -> 亮度調整模式
                    if self.mode_index == 0:  # 固定狀態下不能調亮度
                        return
                    self.in_adjust_mode = True
                    self.adjust_context = "brightness"
                    self.brightness_cursor_index = self.brightness_index
                else:
                    # 調整項目 -> 主頁調整模式
                    self.in_adjust_mode = True
                    self.adjust_context = "page"
                    self.page_cursor_index = self.current_main_index

            elif page == 1:  # 色溫
                if self.sel_color in (0, 1, 2):
                    # 選在 R/G/B -> 選通道+進入該通道比例調整模式
                    if self.mode_index == 0:  # 固定狀態下不能調色溫
                        return
                    self.selected_channel_index = self.sel_color
                    self.in_adjust_mode = True
                    self.adjust_context = "color"
                    self.color_percent_cursor_index = self.channel_percent_indices[self.selected_channel_index]
                else:
                    # 調整項目 -> 主頁調整模式
                    self.in_adjust_mode = True
                    self.adjust_context = "page"
                    self.page_cursor_index = self.current_main_index

            elif page == 2:  # 狀態調整
                if self.sel_status in (0, 1, 2):
                    # 直接切換狀態（按 Enter 才決定）
                    self.mode_index = self.sel_status
                    self._on_mode_changed()
                else:
                    # 調整項目 -> 主頁調整模式
                    self.in_adjust_mode = True
                    self.adjust_context = "page"
                    self.page_cursor_index = self.current_main_index
        else:
            # ===== 調整模式中：Enter = 確認候選值並離開調整模式 =====
            if self.adjust_context == "brightness":
                self.brightness_index = self.brightness_cursor_index
                self._apply_light_state_if_needed()
            elif self.adjust_context == "color":
                self.channel_percent_indices[self.selected_channel_index] = self.color_percent_cursor_index
                self._apply_light_state_if_needed()
            elif self.adjust_context == "page":
                self.current_main_index = self.page_cursor_index

            self.in_adjust_mode = False
            self.adjust_context = None

        self._refresh_ui()

    # ----- 瀏覽模式：只移動「選取子項目」，不改參數 -----
    def _browse_up(self):
        page = self.current_main_index
        if page == 0:
            if self.sel_brightness > 0:
                self.sel_brightness -= 1
        elif page == 1:
            if self.sel_color > 0:
                self.sel_color -= 1
        elif page == 2:
            if self.sel_status > 0:
                self.sel_status -= 1

    def _browse_down(self):
        page = self.current_main_index
        if page == 0:
            if self.sel_brightness < 1:
                self.sel_brightness += 1
            else: 
                self.sel_brightness = 0
        elif page == 1:
            if self.sel_color < 3:
                self.sel_color += 1
            else:
                self.sel_color = 0
        elif page == 2:
            if self.sel_status < 3:
                self.sel_status += 1
            else:
                self.sel_status = 0

    # ----- 調整模式：根據 adjust_context 真正改「候選值」，不立即套用 -----
    def _adjust_up(self):
        # 在「固定」模式時，禁止調整亮度/色溫，但允許切換主頁
        if self.mode_index == 0 and self.adjust_context in ("brightness", "color"):
            return

        if self.adjust_context == "page":
            # 主頁候選左移 (Relax)，循環
            n = len(self.MAIN_PAGES)
            self.page_cursor_index = (self.page_cursor_index - 1) % n

        elif self.adjust_context == "brightness":
            n = len(self.BRIGHTNESS_LEVELS)
            self.brightness_cursor_index = (self.brightness_cursor_index - 1) % n

        elif self.adjust_context == "color":
            n = len(self.COLOR_PERCENT_LEVELS)
            self.color_percent_cursor_index = (self.color_percent_cursor_index - 1) % n

    def _adjust_down(self):
        if self.mode_index == 0 and self.adjust_context in ("brightness", "color"):
            return

        if self.adjust_context == "page":
            # 主頁候選右移 (Concentrate)，循環
            n = len(self.MAIN_PAGES)
            self.page_cursor_index = (self.page_cursor_index + 1) % n

        elif self.adjust_context == "brightness":
            n = len(self.BRIGHTNESS_LEVELS)
            self.brightness_cursor_index = (self.brightness_cursor_index + 1) % n

        elif self.adjust_context == "color":
            n = len(self.COLOR_PERCENT_LEVELS)
            self.color_percent_cursor_index = (self.color_percent_cursor_index + 1) % n

    # ---------- 控制燈 ----------
    def _on_mode_changed(self):
        mode_name = self.MODES[self.mode_index]
        print(f"[INFO] Mode changed to: {mode_name}")

        # 當切換模式時，讓 sel_status 跟著指向目前模式
        self.sel_status = self.mode_index

        if self.mode_index == 1:
            # 熄燈
            self.arduino.set_power(False)
        elif self.mode_index == 2:
            # 調整：開燈並套用目前設定
            self._update_arduino_output()
        elif self.mode_index == 0:
            # 固定：不改變現在的 LED 狀態
            pass

    def _apply_light_state_if_needed(self):
        if self.mode_index == 2:
            self._update_arduino_output()

    def _update_arduino_output(self):
        # 只在「調整」模式下送指令
        if self.mode_index != 2:
            return

        # 色溫 -> base RGB (使用已確定的比例)
        r = int(255 * self.COLOR_PERCENT_LEVELS[self.channel_percent_indices[0]])
        g = int(255 * self.COLOR_PERCENT_LEVELS[self.channel_percent_indices[1]])
        b = int(255 * self.COLOR_PERCENT_LEVELS[self.channel_percent_indices[2]])

        # 整體亮度 -> global brightness (使用已確定的亮度檔位)
        br_ratio = self.BRIGHTNESS_LEVELS[self.brightness_index]
        br_val = int(255 * br_ratio)

        print(f"[INFO] Apply RGB=({r},{g},{b}), BRT={br_val}")
        self.arduino.set_power(True)
        self.arduino.set_color(r, g, b)
        self.arduino.set_brightness(br_val)

    # ---------- 畫面更新 ----------
    def _refresh_ui(self):
        # 更新主功能列 tab 顏色
        for i, lbl in enumerate(self.tab_labels):
            if self.in_adjust_mode and self.adjust_context == "page" and i == self.page_cursor_index:
                bg = self.COLOR_PAGE_CANDIDATE
            elif i == self.current_main_index:
                bg = self.COLOR_PAGE_ACTIVE
            else:
                bg = self.COLOR_PAGE_NORMAL
            lbl.configure(bg=bg)

        # 更新子項目文字 & 顏色
        page = self.current_main_index
        # 先全部清空
        for lbl in self.item_labels:
            lbl.configure(text="", bg=self.COLOR_ITEM_NORMAL)

        if page == 0:
            self._render_brightness_items()
        elif page == 1:
            self._render_color_items()
        else:
            self._render_status_items()

        # 狀態 info
        page_name = self.MAIN_PAGES[self.current_main_index]
        mode_name = self.MODES[self.mode_index]
        selected_item = self._current_selected_item_name()

        if self.in_adjust_mode and self.adjust_context == "page":
            page_info = f"主頁預選: {self.MAIN_PAGES[self.page_cursor_index]}"
        else:
            page_info = ""

        adjust_str = (
            f"調整模式({self.adjust_context})" if self.in_adjust_mode else "瀏覽模式"
        )
        self.info_label_var.set(
            f"目前頁面: {page_name}    燈狀態: {mode_name}    選取: {selected_item}    "
            f"{page_info}    狀態: {adjust_str}"
        )

    def _current_selected_item_name(self) -> str:
        page = self.current_main_index
        if page == 0:
            return "整體亮度" if self.sel_brightness == 0 else "調整項目"
        elif page == 1:
            if self.sel_color < 3:
                return f"{self.COLOR_CHANNELS[self.sel_color]} 通道"
            else:
                return "調整項目"
        elif page == 2:
            if self.sel_status < 3:
                return self.MODES[self.sel_status]
            else:
                return "調整項目"
        return ""

    # --- 子項目 render：用顏色高亮目前所在項目 & 正在調整的項目 ---
    def _render_brightness_items(self):
        # item 0: 整體亮度 (顯示四檔，候選用不同標記)
        if self.in_adjust_mode and self.adjust_context == "brightness":
            active_idx = self.brightness_cursor_index
        else:
            active_idx = self.brightness_index

        parts = []
        for i, label in enumerate(self.BRIGHTNESS_LABELS):
            if i == active_idx:
                parts.append(f"[{label}]")
            else:
                parts.append(label)
        text0 = "整體亮度: " + "  ".join(parts)

        # item 1: 調整項目
        text1 = "調整項目（進入主頁調整模式）"

        # 設文字
        self.item_labels[0].configure(text=text0)
        self.item_labels[1].configure(text=text1)

        # 設顏色
        for idx, lbl in enumerate(self.item_labels[:2]):
            bg = self.COLOR_ITEM_NORMAL
            if idx == self.sel_brightness:
                if self.in_adjust_mode and self.adjust_context == "brightness" and idx == 0:
                    bg = self.COLOR_ITEM_ADJUST
                elif (
                    self.in_adjust_mode and self.adjust_context == "page" and idx == 1
                ):
                    bg = self.COLOR_ITEM_ADJUST
                else:
                    bg = self.COLOR_ITEM_SELECTED
            lbl.configure(bg=bg)

    def _render_color_items(self):
        # R/G/B 三個通道
        for idx, ch in enumerate(self.COLOR_CHANNELS):
            # 顯示比例（候選 or 已確定）
            if (
                self.in_adjust_mode
                and self.adjust_context == "color"
                and self.selected_channel_index == idx
            ):
                perc_idx = self.color_percent_cursor_index
            else:
                perc_idx = self.channel_percent_indices[idx]

            perc_label = self.COLOR_PERCENT_LABELS[perc_idx]
            text = f"{ch} 通道: {perc_label}"
            self.item_labels[idx].configure(text=text)

        # 調整項目
        self.item_labels[3].configure(text="調整項目（進入主頁調整模式）")

        # 設顏色
        for idx, lbl in enumerate(self.item_labels):
            bg = self.COLOR_ITEM_NORMAL
            if idx == self.sel_color:
                # 正在調整該通道
                if (
                    self.in_adjust_mode
                    and self.adjust_context == "color"
                    and idx == self.selected_channel_index
                ):
                    bg = self.COLOR_ITEM_ADJUST
                elif (
                    self.in_adjust_mode
                    and self.adjust_context == "page"
                    and idx == 3
                ):
                    bg = self.COLOR_ITEM_ADJUST
                else:
                    bg = self.COLOR_ITEM_SELECTED
            lbl.configure(bg=bg)

    def _render_status_items(self):
        # 三個狀態
        for i, name in enumerate(self.MODES):
            mark = "[X]" if i == self.mode_index else "[ ]"
            self.item_labels[i].configure(text=f"{mark} {name}")

        # 調整項目
        self.item_labels[3].configure(text="調整項目（進入主頁調整模式）")

        # 設顏色
        for idx, lbl in enumerate(self.item_labels):
            bg = self.COLOR_ITEM_NORMAL
            if idx == self.sel_status:
                if self.in_adjust_mode and self.adjust_context == "page" and idx == 3:
                    bg = self.COLOR_ITEM_ADJUST
                else:
                    bg = self.COLOR_ITEM_SELECTED
            lbl.configure(bg=bg)

    # ---------- 關閉 ----------
    def on_close(self):
        self.stop_event.set()
        try:
            self.arduino.close()
        except Exception:
            pass
        self.root.destroy()


def main():
    # 先載入 SVM 模型（如果不存在就只用鍵盤）
    init_svm_model("bci_svm.pkl")

    root = tk.Tk()
    app = BCILightGUI(root, arduino_port="COM5")  # COM port 改成你自己的
    root.mainloop()



if __name__ == "__main__":
    main()
