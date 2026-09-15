# py

Edge AI experiments on MediaTek Genio: real-time YOLOv8 detection + ReID over
GStreamer/NNstreamer and ONNX Runtime NPU (Neuron EP), plus a carton/date
recognition app (Semicon2026).

- `MediaMTX/` — RTSP client pipelines, YOLOv8 detection + ReID tracking (CPU and Neuron EP variants)
- `NNstreamer/` — GStreamer/NNstreamer inference demos and model assets
- `Semicon2026/` — carton/date recognition app for the Genio 720 viewer, dataset splitting tools

## 如何用指令手動 push 到 GitHub

這台開發板（Yocto 客製化系統）沒有 `git` / `gh` 指令，本 repo 是用 `pip install
dulwich`（純 Python 的 git 實作）push 上去的。如果你是在一般有 `git` 的機器上操作，
流程如下：

### 1. 建立 Personal Access Token

到 https://github.com/settings/tokens → Generate new token (classic) → 勾選
`repo` 權限。

> GitHub 從 2021/8/13 起已關閉「帳號密碼」做 Git HTTPS 認證，push/pull 時的密碼欄位
> 必須換成 token，跟用什麼 git 工具無關。

### 2. 初始化並 commit

```bash
cd /py
git init
git add .
git commit -m "Initial commit"
```

### 3. 設定 remote 並 push

**方式 A：token 直接寫在 URL 裡**（簡單，但 token 會明碼留在 `.git/config`）

```bash
git remote add origin https://<YOUR_TOKEN>@github.com/henryliang3027/py.git
git branch -M main
git push -u origin main
```

**方式 B：push 時系統跳出帳密提示再輸入**（推薦，不會把 token 寫死在 remote URL）

```bash
git remote add origin https://github.com/henryliang3027/py.git
git branch -M main
git push -u origin main
```

跳出提示時：
- Username: 你的 GitHub 帳號（`henryliang3027`）
- Password: 貼上 token（不是 GitHub 登入密碼）

### 4.（可選）快取憑證，之後不用每次貼 token

```bash
git config --global credential.helper store
```

第一次成功後 token 會存在 `~/.git-credentials`（明碼）。

### 注意事項

- Token 就是密碼，不要 commit 進程式碼、不要貼在公開的地方。
- 用不到之後建議到 GitHub token 設定頁 revoke 掉。
- `MediaMTX/resnet50_market1501_aicity156.onnx`（約 92MB）超過 GitHub 建議的
  50MB 上限（但未超過 100MB 硬限制），push 時會看到警告，可正常上傳；如果之後模型
  檔案變得更大，需要改用 Git LFS。
