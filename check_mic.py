# 測試麥克風用的程式
# 需要安裝 sounddevice 和 numpy 套件
import sounddevice as sd
import numpy as np

def list_audio_devices():
    """列出所有可用的音訊裝置。"""
    print("--- 可用的音訊裝置 ---")
    try:
        print(sd.query_devices())
        print("--------------------")
        default_input = sd.default.device[0]
        print(f"預設輸入裝置索引: {default_input}")
    except Exception as e:
        print(f"查詢裝置時發生錯誤: {e}")
        print("可能是系統中沒有任何可辨識的錄音裝置。")

def test_microphone():
    """測試麥克風輸入並顯示音量條。"""
    print("\n--- 開始測試麥克風 ---")
    print("請對著麥克風說話，你應該會看到下方的音量條變化。")
    print("按下 Ctrl+C 即可結束測試。")

    try:
        def print_volume(indata, frames, time, status):
            """這個回呼函式會被音訊串流持續呼叫。"""
            if status:
                print(status)
            # 計算音量大小（均方根）
            volume_norm = np.linalg.norm(indata) * 10
            # 建立一個簡單的視覺化音量條
            bar = '█' * int(volume_norm)
            print(f"音量: |{bar:<50}|", end='\r')

        # 使用 InputStream 持續監聽麥克風
        with sd.InputStream(callback=print_volume, samplerate=16000, channels=1):
            while True:
                # 讓程式保持執行，直到使用者按下 Ctrl+C
                sd.sleep(1000)

    except KeyboardInterrupt:
        print("\n--- 麥克風測試結束 ---")
    except Exception as e:
        print(f"\n開啟麥克風串流時發生錯誤: {e}")
        print("請檢查你的麥克風是否已連接，且未被其他程式佔用。")
        print("同時，請確認作業系統已授予此應用程式存取麥克風的權限。")

if __name__ == "__main__":
    list_audio_devices()
    test_microphone()