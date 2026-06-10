# EHS CCTV Monitoring — Hướng dẫn sử dụng

Ứng dụng giám sát an toàn lao động (EHS): xem nhiều camera, phát hiện vi phạm bằng AI (YOLO), hiển thị cảnh báo và lưu ảnh sự cố.

**Phiên bản khuyến nghị:** chạy từ thư mục `ver2/` (giao diện tab Monitor + Settings, lưu cấu hình hiển thị trong `settings.json`).

---

## Mục lục

1. [Cài đặt](#cài-đặt)
2. [Chạy ứng dụng lần đầu](#chạy-ứng-dụng-lần-đầu)
3. [Giao diện](#giao-diện)
4. [Tab Monitor](#tab-monitor)
5. [Tab Settings](#tab-settings)
6. [Hai file cấu hình](#hai-file-cấu-hình)
7. [Cấu hình camera (`config_data.json`)](#cấu-hình-camera-config_datajson)
8. [Cài đặt vận hành (`settings.json`)](#cài-đặt-vận-hành-settingsjson)
9. [Ảnh cảnh báo & panel log](#ảnh-cảnh-báo--panel-log)
10. [Cài FFmpeg (RTSP)](#cài-ffmpeg-rtsp)
11. [Xử lý sự cố thường gặp](#xử-lý-sự-cố-thường-gặp)

---

## Cài đặt

### Yêu cầu

| Thành phần | Ghi chú |
|------------|---------|
| Windows 10/11 | |
| Python 3.9+ | Khuyến nghị 3.10 hoặc 3.11 |
| PyQt5, OpenCV, Ultralytics, PyTorch | Xem lệnh cài bên dưới |
| FFmpeg | **Bắt buộc nếu dùng RTSP** — phải có trong PATH |
| File model `.pt` | Ví dụ `yolo11n.pt` đặt trong `ver2/` |

### Cài thư viện Python

```powershell
pip install PyQt5 opencv-python ultralytics torch unidecode pillow
```

GPU NVIDIA: cài PyTorch bản CUDA từ [pytorch.org](https://pytorch.org/).

### Kiểm tra FFmpeg

```powershell
ffmpeg -version
```

Nếu lỗi “không tìm thấy lệnh”, xem [Cài FFmpeg](#cài-ffmpeg-rtsp).

---

## Chạy ứng dụng lần đầu

```powershell
cd D:\2025\CCTV\ver2
python Program.py
```

**Trước khi chạy:**

1. Chỉnh đường dẫn camera trong `ver2/config_data.json` (`camera_src`).
2. Đặt file model (ví dụ `yolo11n.pt`) đúng đường dẫn trong config.
3. Lần đầu chưa có `settings.json` → app tự tạo, mặc định hiển thị 4 camera đầu (CCTV 01–04).

> **Lưu ý:** Bản cũ ở thư mục gốc (`python Program.py` không qua `ver2`) vẫn chạy được nhưng không có tab Settings / `settings.json`. Nên dùng `ver2/`.

---

## Giao diện

Cửa sổ gồm **hai tab**:

| Tab | Mục đích |
|-----|----------|
| **Monitor** | Xem trực tiếp tối đa **4 camera** + panel ảnh cảnh báo bên phải |
| **Settings** | Chọn camera hiển thị, lưu cài đặt vào `settings.json` |

Hệ thống có thể khai báo **tối đa 8 camera** trong `config_data.json`, nhưng màn hình Monitor chỉ hiển thị **4 camera** do bạn chọn trong Settings.

---

## Tab Monitor

### Bố cục

- **Bên trái:** lưới 2×2 (4 ô camera).
- **Bên phải:** ảnh cảnh báo mới nhất + danh sách file trong `LastDetectionWarning/`.

### Trạng thái mỗi ô camera

| Hiển thị | Ý nghĩa |
|----------|---------|
| **LIVE** (chấm xanh) | Đang nhận frame từ nguồn video |
| **NO SIGNAL** (vàng) | Mất tín hiệu / không đọc được frame |
| **OFF** / **CHƯA CHỌN** | Ô chưa gán camera hoặc stream đang tắt |
| **WARNING** | Phát hiện vi phạm (không mũ / ngã / không áo theo rule đã bật) |

### Checkbox trên từng ô

Bật/tắt rule phát hiện **ngay trên màn hình** (áp dụng cho phiên làm việc hiện tại):

- **Helmet** — cảnh báo liên quan mũ bảo hộ  
- **Fell** — cảnh báo ngã  
- **Jacket** — cảnh báo áo bảo hộ  

Rule mặc định lấy từ `config_data.json` → `enable_flags`.

### Hình ảnh có khung box

- **Đỏ:** vi phạm (theo rule đang bật).  
- **Xanh:** đối tượng phát hiện nhưng không thuộc rule cảnh báo.

---

## Tab Settings

Dùng khi cần **đổi bộ camera hiển thị** trên Monitor (trong tối đa 8 camera đã khai báo).

### Quy trình

1. Mở tab **Settings** → tất cả luồng trên Monitor **tạm dừng** (tránh xung đột RTSP).
2. Tick **Hiển thị** tối đa **4 camera** trong danh sách.
3. Chọn một trong hai nút:

| Nút | Tác dụng |
|-----|----------|
| **Lưu cài đặt** | Ghi `settings.json` — **không** mở lại stream |
| **Áp dụng và về Monitor** | Ghi `settings.json` + gán camera vào 4 ô + quay tab Monitor và **bật stream** |

4. Lần mở app sau → đọc `settings.json` → tự hiển thị đúng bộ camera đã lưu.

### Ví dụ

Muốn xem CCTV 01, 02, 05, 06 thay vì 01–04:

1. Settings → bỏ tick CCTV 03, 04.  
2. Tick CCTV 05, 06.  
3. Bấm **Áp dụng và về Monitor**.

---

## Hai file cấu hình

```text
config_data.json   →  Danh mục camera (RTSP, model, rule detect) — ít đổi
settings.json      →  Camera nào đang hiển thị + tham số vận hành — đổi qua UI
config_data_log.json → Panel ảnh cảnh báo bên phải
```

| Câu hỏi | File nào |
|---------|----------|
| Thêm camera mới / đổi URL RTSP? | `config_data.json` |
| Chọn 4 camera hiển thị hôm nay? | Tab Settings → `settings.json` |
| Bật/tắt rule helmet/fell mặc định? | `config_data.json` → `enable_flags` |

**Không** dùng `use_camera` trong `config_data.json` nữa — việc “camera nào lên màn hình” nằm trong `settings.json`.

---

## Cấu hình camera (`config_data.json`)

File nằm tại `ver2/config_data.json`. Mỗi phần tử trong `camera_infos` là một camera.

### Các trường quan trọng

| Trường | Mô tả | Ví dụ |
|--------|--------|--------|
| `camera_name` | Tên hiển thị (khớp với `settings.json`) | `"CCTV 01"` |
| `camera_src` | RTSP URL, đường dẫn file video, hoặc `0` (USB) | `rtsp://user:pass@ip/...` |
| `yolo_model_path` | File weight YOLO | `yolo11n.pt` |
| `img_size` | Kích thước input AI | `640` |
| `yolo_rate` | Ngưỡng confidence (0.0–1.0) | `0.5` |
| `roi_check` | Vùng quan tâm `[x1, x2, y1, y2]` | `[-1, 9999, -1, 9999]` = toàn khung |
| `enable_flags` | Rule detect mặc định | xem bảng dưới |

### `enable_flags` (chỉ rule phát hiện)

```json
"enable_flags": {
  "fell": 1,
  "helmet": 1,
  "jacket": 0,
  "fire": 0,
  "smoke": 0
}
```

`1` = bật, `0` = tắt.

### Thêm camera thứ 5–8

Thêm object mới vào mảng `camera_infos`, đặt `camera_name` **duy nhất**. Camera mới **không** tự hiện trên Monitor — chọn trong tab Settings rồi Lưu/Áp dụng.

Sau khi sửa `config_data.json`, **khởi động lại** ứng dụng (hoặc vào Settings → Áp dụng lại nếu chỉ đổi tên camera đã có trong settings).

---

## Cài đặt vận hành (`settings.json`)

File tại `ver2/settings.json`, tạo tự động hoặc chỉnh qua tab Settings.

### Ví dụ

```json
{
  "version": 1,
  "display": {
    "max_active_cameras": 4,
    "monitor_columns": 2,
    "selected_cameras": [
      "CCTV 01",
      "CCTV 02",
      "CCTV 03",
      "CCTV 04"
    ]
  },
  "performance": {
    "display_target_fps": 6.0,
    "swap_delay_ms": 800
  }
}
```

| Trường | Ý nghĩa |
|--------|---------|
| `selected_cameras` | Danh sách tên camera hiển thị (thứ tự = Ô 1, 2, 3, 4) |
| `max_active_cameras` | Số ô Monitor (mặc định 4) |
| `display_target_fps` | Tần suất gửi frame vào AI mỗi camera (~6 = ổn định) |
| `swap_delay_ms` | Dự phòng cho đổi cam (ms) — dùng khi mở rộng sau |

**Chỉnh tay:** có thể sửa `settings.json` khi app **đang tắt**, rồi mở lại. Tên trong `selected_cameras` phải trùng `camera_name` trong `config_data.json`.

---

## Ảnh cảnh báo & panel log

- Khi phát hiện vi phạm, ảnh được lưu vào:  
  `ver2/LastDetectionWarning/`  
  Tên file: `{tên_camera}_{thời_gian}_warning.jpg`
- Panel bên phải tab Monitor đọc thư mục này (cấu hình `config_data_log.json` → `directory`).

---

## Cài FFmpeg (RTSP)

RTSP chạy ổn định hơn khi có FFmpeg trong PATH.

### Cài nhanh (winget)

```powershell
winget install --id Gyan.FFmpeg
```

Đóng/mở lại terminal, kiểm tra `ffmpeg -version`.

### Tải thủ công

1. Tải bản Windows từ [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/) (bản *essentials*).  
2. Giải nén, thêm thư mục `bin` vào **Path** (ví dụ `C:\ffmpeg\bin`).  

### Test RTSP

```powershell
ffmpeg -rtsp_transport tcp -i "rtsp://user:pass@ip/..." -t 5 -f null -
```

---

## Xử lý sự cố thường gặp

| Triệu chứng | Cách xử lý |
|-------------|------------|
| **NO SIGNAL** liên tục | Kiểm tra `camera_src`, mạng, cài FFmpeg; test RTSP bằng lệnh trên |
| Không thấy camera sau khi chọn Settings | Bấm **Áp dụng và về Monitor**, không chỉ Lưu |
| Camera trong settings không chạy | Kiểm tra `camera_name` khớp chính xác với config (kể cả khoảng trắng) |
| App chậm / giật | Giảm `display_target_fps` trong `settings.json` (vd `4.0`); giảm `img_size` xuống `512` |
| Quá nhiều cảnh báo sai | Tăng `yolo_rate` trong config (vd `0.6`) |
| Không load được model | Đặt đúng đường dẫn `yolo_model_path`; chạy app từ thư mục `ver2` |
| Log `YOLO queue overloaded` | Giảm FPS hoặc số camera đang hiển thị |

---

## Cấu trúc thư mục (ver2)

```text
ver2/
├── Program.py              ← Chạy ứng dụng
├── config_data.json        ← Khai báo camera
├── settings.json           ← Camera hiển thị + FPS (Settings tab)
├── config_data_log.json    ← Panel log ảnh
├── CameraWidget.py         ← Ô camera + capture
├── yolo_engine.py          ← AI dùng chung
├── camera_settings_panel.py← UI tab Settings
├── settings_store.py       ← Đọc/ghi settings.json
├── ffmpeg_capture.py       ← RTSP qua FFmpeg
└── LastDetectionWarning/   ← Ảnh vi phạm (tự tạo)
```

---

## Tóm tắt quy trình hàng ngày

1. Mở app: `cd ver2` → `python Program.py`.  
2. Tab **Monitor**: quan sát 4 camera, xử lý khi thấy **WARNING**.  
3. Cần đổi camera hiển thị → tab **Settings** → chọn tối đa 4 → **Áp dụng và về Monitor**.  
4. Cần thêm/sửa RTSP hoặc model → sửa `config_data.json` → khởi động lại app.  
5. Xem lại sự cố → panel phải hoặc thư mục `LastDetectionWarning/`.
