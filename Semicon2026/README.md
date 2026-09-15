# Genio 720 Viewer

## SSH 登入後的必要設定

用 root 透過 SSH 連進去執行 `image_receive.py`（或 `image_receive_gen_data.py`）前，要先 export `XDG_RUNTIME_DIR`，否則 GTK 會噴：

```
RuntimeError: Gtk couldn't be initialized. Use Gtk.init_check() if you want to handle this case.
```

原因：畫面是由 weston（Wayland compositor）提供，socket 在 `/run/wayland-0`（root 屬於 `wayland` group 所以能存取）。但 SSH 進來的 shell 預設沒有 `XDG_RUNTIME_DIR`，GDK 找不到 runtime 目錄就連不上 compositor。

`/run/user/1000` 是 `weston` 使用者自己的 runtime dir（權限 `drwx------`），root 沒有權限，**不能用**。

每次 SSH 進去後先執行：

```bash
export XDG_RUNTIME_DIR=/run
```

`WAYLAND_DISPLAY` 不用另外設，預設值 `wayland-0` 剛好對應 `/run/wayland-0`。

（`~/.bashrc` 裡已經加了這行 export，正常新開的 shell 會自動生效；如果沒生效再手動執行一次。）

## `/infer` API（供 Android app 串接）

`POST http://<device-ip>:5000/infer`，multipart form-data，欄位名稱 `image`（任意 JPEG/PNG）。

Request 會同步跑完兩階段偵測（紙箱 → 紙箱內的日期印刷區）後才回應，回傳內容是 JSON，同時畫面上的 GTK 預覽也會更新。

### Response schema

```json
{
  "image_width": 1920,
  "image_height": 1080,
  "boxes": [
    {
      "label": "維他露P",
      "score": 0.8734,
      "bbox": [120.5, 340.2, 560.8, 900.1],
      "date_str": "2027年12月22日",
      "date_bbox": [230.1, 400.5, 350.2, 430.9],
      "date_score": 0.8631
    },
    {
      "label": "樂事洋芋片青檸口味",
      "score": 0.7521,
      "bbox": [900.0, 150.3, 1400.2, 700.9],
      "date_str": "2027年1月13日",
      "date_bbox": null,
      "date_score": null
    }
  ]
}
```

欄位說明：

| 欄位 | 型別 | 說明 |
|---|---|---|
| `image_width` / `image_height` | int | 上傳圖片的原始像素尺寸（EXIF 方向校正後）。所有 bbox 座標都是相對於這個尺寸。 |
| `boxes` | array | 偵測到的紙箱清單。目前只回傳 `TARGET_BOX_LABELS`（`維他露P`、`樂事洋芋片青檸口味`）這兩類，其餘 11-class 模型偵測到的類別會被過濾掉。 |
| `boxes[].label` | string | 紙箱類別名稱。 |
| `boxes[].score` | float | 紙箱偵測信心值 (0~1)。 |
| `boxes[].bbox` | `[x1, y1, x2, y2]` | 紙箱框，**原圖座標**（左上/右下角，單位 px）。 |
| `boxes[].date_str` | string \| null | 依 `DATE_STR_BY_BOX_LABEL` 對照表給的固定日期字串；若該 `label` 不在對照表中則為 `null`。 |
| `boxes[].date_bbox` | `[x1, y1, x2, y2]` \| null | 該紙箱內偵測到的日期印刷區框，**原圖座標**（已把 crop 內的相對座標換算回原圖 offset）。若這個紙箱裡沒偵測到日期區則為 `null`。 |
| `boxes[].date_score` | float \| null | 日期區偵測信心值；沒偵測到時為 `null`。 |

備註：
- 若同一個紙箱的 crop 裡偵測到多個日期候選框，只保留信心值最高的一個。
- 空陣列 `boxes: []` 代表沒偵測到任何目標類別的紙箱。

## `split.py`（切分資料集並打包給 Roboflow）

把本地訓練用的 YOLO 格式資料集切成 train/valid，轉成 Roboflow 匯入所需的目錄結構與 zip 檔。

處理兩組資料集：`box`（11 類別紙箱偵測）與 `date`（1 類別日期偵測）。

- 來源：`for_training/<box|date>/{images,labels}`
- 輸出：`for_roboflow_split/<box|date>/{images,labels}/{train,val}`

流程：
1. 讀取來源資料夾內所有圖片（`.jpg/.jpeg/.png/.bmp`），用固定 seed 洗牌後依 `train_ratio` 切成 train/val 兩堆，並複製對應的 YOLO label（`.txt`）；找不到對應 label 時只印警告，圖片仍會複製過去。
2. （預設會做）把切好的 `box/` 、`date/` 資料夾各自打包成 `box.zip` / `date.zip`，並把對應的 label template（`label_template_box_11cls.txt` / `label_template_date_1cls.txt`）一併放進 zip 根目錄。

用法：

```bash
python split.py                          # 用預設值：train_ratio=0.8, seed=42，並打包成 zip
python split.py --train-ratio 0.9        # 調整切分比例
python split.py --seed 123               # 調整洗牌 seed（影響切分結果的可重現性）
python split.py --zip false              # 只切分，不打包 zip
```
