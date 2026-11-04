# 薩克斯風移調器

上傳 JPG／PDF 曲譜或貼上簡譜，轉成高音、中音、次中音、上低音薩克斯風的記譜，檢查音域，並下載 PDF 曲譜。

## 檔案

| 檔案 | 用途 |
| --- | --- |
| `streamlit_app.py` | 主程式（Streamlit 介面） |
| `transpose.py` | 移調與音域計算 |
| `score_reader.py` | 用 Claude 讀取曲譜圖片／PDF |
| `score_pdf.py` | 產生 A4 簡譜 PDF |
| `requirements.txt` | 需要的套件 |
| `.streamlit/config.toml` | 主題顏色、上傳大小上限 |
| `.streamlit/secrets.toml.example` | Secrets 範例（不要上傳真正的 secrets.toml） |

## 部署到 Streamlit Community Cloud（免費）

1. **建立 GitHub 儲存庫**：到 <https://github.com/new>，取個名字（例如 `sax-transposer`），選 Public，按 Create repository。
2. **上傳檔案**：在新的儲存庫頁面點 “uploading an existing file”，把上表的檔案拖進去。`.streamlit` 資料夾要整個拖進去（它是隱藏資料夾，Windows 檔案總管要開「顯示隱藏的項目」才看得到）。按 Commit changes。
3. **建立 App**：到 <https://share.streamlit.io>，用 GitHub 帳號登入，按 Create app →「Deploy a public app from GitHub」：
   - Repository：剛剛的儲存庫
   - Branch：`main`
   - Main file path：`streamlit_app.py`
   - App URL：可以自訂成好記的網址，例如 `sax-transposer.streamlit.app`
4. **Advanced settings**：Python version 選 3.12，Secrets 保持空白。
5. 按 Deploy。第一次安裝套件約需 2–5 分鐘，完成後網址就固定不變。

## API key 與密碼

- **預設**：Secrets 留空，不需要密碼，每個人在側欄輸入自己的 Anthropic API key。手動移調與 PDF 下載不需要 key。
- **選用**：若想把自己的 key 存在 Cloud 上，見 `secrets.toml.example`；這時一定要同時設 `APP_PASSWORD`，否則任何拿到網址的人都能用你的 key。

## 注意

- 免費方案的 App 若 7 天以上沒人使用會進入休眠，下次開啟時按 “Yes, get this app back up!” 等約 30 秒即可，網址不變。
- 之後修改程式，只要把新檔案上傳到 GitHub 同一個儲存庫，App 會自動更新。

## 在自己電腦執行

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```
