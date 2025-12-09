# client_A_with_focus_low_latency.py
import socket
import time
import pyautogui
import pyperclip
import pygetwindow as gw

SERVER_IP = "192.168.50.187"  # 換成自己的IP
SERVER_PORT = 50007
RECORD_SECONDS = 3.0
SLEEP_AFTER_HOTKEY = 0.3
SLEEP_AFTER_CONVERT = 0.8
RETRY_IF_NO_WINDOW = 5

BIOSAC_WINDOW_KEYWORD = "Biopac Student Lab"

pyautogui.FAILSAFE = False


def activate_biosac_window() -> bool:
    windows = gw.getWindowsWithTitle(BIOSAC_WINDOW_KEYWORD)
    if not windows:
        print(f"[Client] 找不到包含「{BIOSAC_WINDOW_KEYWORD}」標題的視窗。")
        return False

    win = windows[0]
    if win.isMinimized:
        win.restore()
    win.activate()
    time.sleep(0.3)
    print(f"[Client] 已將視窗「{win.title}」叫到最前面。")
    return True


def main():
    print(f"[Client] Connecting to {SERVER_IP}:{SERVER_PORT} ...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # === 低延遲相關設定（新增） ===
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)      # 關掉 Nagle
        # 可選：調整 buffer，根據需要決定要不要
        # sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024)
        # sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024)
        # ============================

        sock.connect((SERVER_IP, SERVER_PORT))
        print("[Client] Connected.")
        print("[Client] 程式會自動尋找並切換到 BIOSAC 視窗，按 Ctrl+C 可停止。\n")

        while True:
            if not activate_biosac_window():
                print(f"[Client] {RETRY_IF_NO_WINDOW} 秒後再試一次...\n")
                time.sleep(RETRY_IF_NO_WINDOW)
                continue

            print("[Client] Start recording...")
            pyautogui.hotkey("ctrl", "space")
            time.sleep(RECORD_SECONDS)

            print("[Client] Stop recording.")
            pyautogui.hotkey("ctrl", "space")
            time.sleep(SLEEP_AFTER_HOTKEY)

            pyautogui.hotkey("ctrl", "a")
            time.sleep(SLEEP_AFTER_HOTKEY)

            pyautogui.hotkey("ctrl", "l")
            time.sleep(SLEEP_AFTER_CONVERT)

            pyautogui.hotkey("ctrl", "x")
            time.sleep(SLEEP_AFTER_HOTKEY)

            text = pyperclip.paste()
            if not text.strip():
                print("[Client] Clipboard is empty, skip sending.\n")
                continue

            data = text.encode("utf-8", errors="replace")
            sock.sendall(data)
            print(f"[Client] Sent {len(data)} bytes to server.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Client] Stopped by user.")
