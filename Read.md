# Google Gemini Live API 語音對話客戶端
主要功能包括:
- 使用 websockets 連接到Gemini Live API
- 串流語音
- 保存輸出音頻到 .WAV 文件

# 主要函式:
- AudioConfig: 音頻流配置數據類
- RunningLiveAudio: 使用sounddevice管理實時音頻輸入/輸出
- 各種編碼/解碼函數: 處理API通信的數據格式
- main(): 主異步循環，協調音頻流和API通信

# 使用要求:
- 使用虛擬環境(建議)，我個人使用 venv + python 3.12
- 需要在.env文件中設置GOOGLE_API_KEY
- use reqirements.txt to install 模組

# 範例使用方式:
使用 check_mic.py 測試麥克風
使用指令執行: python streaming.py

# 範例專案:
Colab示例: https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/websockets/LiveAPI_streaming_in_colab.ipynb

