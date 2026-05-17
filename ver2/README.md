# EHS CCTV Monitoring

Ứng dụng giám sát an toàn lao động (EHS) trên nhiều camera đồng thời. Hệ thống đọc luồng video (RTSP, file, USB), chạy phát hiện đối tượng bằng YOLO trên GPU/CPU, và hiển thị cảnh báo trực tiếp trên giao diện PyQt5.

Thiết kế tối ưu cho **4–6 camera** chạy ổn định 24/7: một engine YOLO dùng chung, capture RTSP qua FFmpeg, throttle FPS từng camera để tránh quá tải GPU.

---

## Mục lục

1. [Tính năng](#tính-năng)
2. [Kiến trúc](#kiến-trúc)
3. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
4. [Yêu cầu & cài đặt](#yêu-cầu--cài-đặt)
5. [Chạy ứng dụng](#chạy-ứng-dụng)
6. [Cấu hình](#cấu-hình)
7. [Luồng xử lý](#luồng-xử-lý)
8. [Tuning hiệu năng](#tuning-hiệu-năng)
9. [Cài đặt FFmpeg](#cài-đặt-ffmpeg)
10. [Log & ảnh cảnh báo](#log--ảnh-cảnh-báo)
11. [Xử lý sự cố](#xử-lý-sự-cố)

---

## Tính năng

- Giám sát **nhiều camera** trên một màn hình (grid tự động theo số camera bật).
- Phát hiện vi phạm EHS: **không đội mũ (helmet)**, **ngã (fell)**, **không mặc áo (jacket)** — bật/tắt theo từng camera.
- Nguồn video: **RTSP**, file video, hoặc chỉ số camera USB.
- RTSP ổn định qua **FFmpeg** (TCP, auto-reconnect, backoff).
- **Một model YOLO** dùng chung cho tất cả camera (cache theo đường dẫn weight).
- Batch inference tối đa 4 frame/lần khi các camera dùng cùng cấu hình model.
- Overlay **WARNING** trên tile camera, lưu ảnh vi phạm vào thư mục `LastDetectionWarning/`.
- Panel xem lại ảnh cảnh báo (cấu hình qua `config_data_log.json`).

---

## Kiến trúc

```mermaid
flowchart TB
  subgraph config [Cấu hình]
    CD[config_data.json]
    CL[config_data_log.json]
  end

  subgraph app [Ứng dụng PyQt5]
    P[Program.py]
    MW[MainWindow]
    CW[CameraWidget × N]
    LW[LogWidget + ImageListWidget]
  end

  subgraph capture [Capture — mỗi camera]
    VC[VideoCapture]
    FF[FFmpegCapture]
    CV[OpenCV fallback]
  end

  subgraph infer [Inference — dùng chung]
    YE[YoloEngine]
    Q[Queue maxsize=16]
    W[Worker thread]
    B[Batch predict ≤4]
  end

  CD --> P --> MW
  CL --> MW
  MW --> CW
  MW --> LW
  CW --> VC
  VC --> FF
  VC --> CV
  CW -->|throttle ~6 FPS| YE
  YE --> Q --> W --> B
  B -->|result_ready| CW
  B --> DISK[LastDetectionWarning/]
```

| Thành phần | File | Vai trò |
|------------|------|---------|
| Entry point | `Program.py` | Khởi tạo UI, load config, tạo grid camera |
| Widget camera | `CameraWidget.py` | Capture frame, gửi YOLO, hiển thị kết quả |
| Engine YOLO | `yolo_engine.py` | Cache model, queue job, batch predict, emit kết quả |
| Capture RTSP | `ffmpeg_capture.py` | Đọc RTSP qua subprocess FFmpeg |
| Cấu hình | `config_data.py` | Parse `config_data.json` |
| Log | `Logging.py` | Ghi log ứng dụng |

**Điểm khác so với kiến trúc cũ:** mỗi camera không còn load riêng một model YOLO và không còn QThread inference riêng. Toàn bộ inference đi qua `YoloEngine` — giảm VRAM và tránh chia nhỏ GPU context.

---

## Cấu trúc thư mục

```
CCTV/
├── Program.py              # Chạy ứng dụng
├── CameraWidget.py         # UI + capture từng camera
├── yolo_engine.py          # YOLO engine dùng chung
├── ffmpeg_capture.py       # RTSP capture qua FFmpeg
├── config_data.json        # Cấu hình camera chính
├── config_data_log.json    # Cấu hình panel log/ảnh cảnh báo
├── config_data.py          # Loader config
├── display_image.py        # Widget xem ảnh log
├── list_widget.py          # Danh sách ảnh cảnh báo
├── Logging.py              # Logger
├── scale_test_helper.py    # Helper test nhiều camera (tùy chọn)
├── yolo_training.py        # Huấn luyện model (tùy chọn)
├── data.yaml               # Dataset YOLO (tùy chọn)
├── LastDetectionWarning/   # Ảnh vi phạm (tự tạo khi chạy)
├── CodeTest/               # Script thử nghiệm (không dùng production)
└── Code_base_not_log/      # Bản backup cũ (tham khảo)
```

---

## Yêu cầu & cài đặt

### Phần mềm

| Thành phần | Ghi chú |
|------------|---------|
| Python 3.9+ | Khuyến nghị 3.10 hoặc 3.11 |
| PyQt5 | Giao diện |
| OpenCV (`cv2`) | Đọc frame, ghi ảnh |
| Ultralytics | YOLO inference |
| PyTorch | GPU (CUDA) hoặc CPU |
| FFmpeg | **Khuyến nghị** cho RTSP 24/7 — phải có trong `PATH` |
| `unidecode` | Tên file an toàn trên Windows |

### Cài dependency (ví dụ)

```powershell
pip install PyQt5 opencv-python ultralytics torch unidecode pillow
```

Nếu dùng GPU NVIDIA, cài PyTorch bản CUDA phù hợp từ [pytorch.org](https://pytorch.org/).

### File model

Đặt file weight (ví dụ `yolo11n.pt`) trong thư mục project hoặc chỉ đường dẫn đầy đủ trong `config_data.json` → `yolo_model_path`.

---

## Chạy ứng dụng

```powershell
cd D:\2025\CCTV
python Program.py
```

Trước khi chạy:

1. Chỉnh `config_data.json` (nguồn camera, model, rule cảnh báo).
2. Đảm bảo `ffmpeg -version` chạy được nếu dùng RTSP.
3. Với file video test, dùng đường dẫn tuyệt đối hoặc tương đối hợp lệ trong `camera_src`.

---

## Cấu hình

### `config_data.json` — camera giám sát

Mỗi phần tử trong `camera_infos` mô tả một camera:

| Trường | Kiểu | Mô tả |
|--------|------|--------|
| `camera_name` | string | Tên hiển thị trên UI |
| `camera_src` | string / int | RTSP URL, đường dẫn file, hoặc index USB (`0`, `1`, …) |
| `img_size` | int | Kích thước input YOLO (thường `640`) |
| `yolo_model_path` | string | Đường dẫn file `.pt` |
| `yolo_rate` | float | Ngưỡng confidence (0.0–1.0) |
| `classes` | array | Tên class (hiển thị trên label box) |
| `colors` | array | Màu box: `[R,G,B]` hoặc `"#RRGGBB"` |
| `roi_check` | `[x1, x2, y1, y2]` | Vùng quan tâm; box ngoài ROI bị bỏ qua |
| `enable_flags` | object | Bật/tắt camera và rule detect |
| `timer_delay` | int | Delay panel log (ms) |

**`enable_flags`:**

| Cờ | Giá trị | Ý nghĩa |
|----|---------|---------|
| `use_camera` | `1` / `0` | Hiển thị và xử lý camera này |
| `helmet` | `1` / `0` | Cảnh báo không đội mũ |
| `fell` | `1` / `0` | Cảnh báo ngã |
| `jacket` | `1` / `0` | Cảnh báo không mặc áo |
| `fire` | `1` / `0` | (dự phòng) |
| `smoke` | `1` / `0` | (dự phòng) |

**Class ID trong model** (hardcode trong `yolo_engine.py` — phải khớp model đã train):

| Rule | Class index (`cls`) |
|------|---------------------|
| Helmet | `5` |
| Jacket | `8` |
| Fell | `9` |

Nếu đổi model hoặc dataset, cần cập nhật các index này trong `yolo_engine.py` cho đúng `model.names`.

**Ví dụ cấu hình một camera:**

```json
{
  "roi_check": [-1, 9999, -1, 9999],
  "camera_name": "CCTV 01",
  "camera_src": "rtsp://user:pass@192.168.1.100/stream1",
  "img_size": 640,
  "yolo_model_path": "yolo11n.pt",
  "yolo_rate": 0.5,
  "classes": ["person", "bicycle", "car", "motorcycle", "jacket", "fell", "helmet", "truck", "fire", "smoke"],
  "colors": ["#0000FF", "#00FF00", "#FF00FF", "#000000", "#000000",
             "#FF0000", "#FF0000", "#FF0000", "#FF0000", "#FF0000"],
  "enable_flags": {
    "use_camera": 1,
    "fell": 1,
    "helmet": 1,
    "jacket": 0,
    "fire": 0,
    "smoke": 0
  },
  "timer_delay": 100
}
```

`roi_check = [-1, 9999, -1, 9999]` nghĩa là không giới hạn ROI (toàn khung hình).

### `config_data_log.json` — panel ảnh cảnh báo

Cấu hình widget bên phải: xem ảnh mới nhất và danh sách file trong `directory` (mặc định `./LastDetectionWarning/`).

---

## Luồng xử lý

### 1. Khởi động

`Program.py` → `MainWindow` đọc `config_data.json` → lọc camera có `enable_flags.use_camera == 1` → tạo grid `CameraWidget` (số cột ≈ `ceil(sqrt(n))`).

### 2. Capture (mỗi camera)

- `CameraWidget` tạo `VideoCapture` trong thread nền.
- **RTSP + có FFmpeg trong PATH** → `FFmpegCapture`: TCP, timeout, auto-restart, queue `maxsize=1` (chỉ giữ frame mới nhất).
- **Ngược lại** → OpenCV `cv2.VideoCapture` + reconnect exponential backoff.
- UI hiển thị trạng thái **LIVE** / **NO SIGNAL** (timeout 30 giây không có frame).

### 3. Gửi frame sang YOLO

- `QTimer` ~30 ms đọc frame từ queue capture.
- Chỉ gửi inference khi:
  - Không có job đang chờ (`_infer_in_flight == False`), và
  - Đã qua khoảng `_infer_interval` (mặc định `_target_fps = 6.0` → ~6 lần/giây/camera).

### 4. YoloEngine (dùng chung)

- Job vào queue (`maxsize=16`); queue đầy → **drop frame cũ**, ưu tiên frame mới.
- Worker gom batch tối đa **4** job cùng `model_path`, `img_size`, `yolo_rate`.
- Model load **một lần** theo path; CUDA + half precision nếu có GPU.
- Lọc box theo ROI, confidence, và `enable_flags`.
- Box vi phạm: viền đỏ + khung cảnh báo; lưu ảnh tối đa 1 lần / 5 giây / camera.
- Emit `result_ready(camera_name, QImage, is_warning, meta)`.

### 5. Hiển thị UI

- `CameraWidget` lọc signal theo `camera_name`.
- Scale ảnh `KeepAspectRatio` + `SmoothTransformation`.
- Overlay **WARNING** có cooldown 10 phút giữa các lần hiện (tránh nhấp nháy liên tục).

---

## Tuning hiệu năng

Khuyến nghị cho **4–6 camera RTSP** chạy 24/7:

### 1. Tần số inference mỗi camera

| File | Biến | Mặc định | Gợi ý |
|------|------|----------|--------|
| `CameraWidget.py` | `_target_fps` | `6.0` | 4 cam: `8–10`; 6 cam: `6–8` |

Tăng → mượt hơn, GPU nặng hơn. Giảm → bền hơn, ít drop frame.

### 2. Kích thước input YOLO

| File | Trường | Gợi ý |
|------|--------|--------|
| `config_data.json` | `img_size` | `640` cân bằng; `512` nhẹ hơn |

### 3. Ngưỡng confidence

| File | Trường | Gợi ý |
|------|--------|--------|
| `config_data.json` | `yolo_rate` | `0.5–0.6` cân bằng; `0.6–0.7` ít false-positive |

### 4. Batch inference

| File | Biến | Mặc định | Gợi ý |
|------|------|----------|--------|
| `yolo_engine.py` | `_batch_size` | `4` | Throughput: `4`; latency thấp: `2–3` |

### 5. Queue inference

| File | Biến | Mặc định | Nguyên tắc |
|------|------|----------|------------|
| `yolo_engine.py` | `Queue(maxsize=…)` | `16` | Nhỏ → latency thấp khi quá tải; lớn → chịu spike nhưng dễ trễ |

### 6. Capture RTSP

| File | Tham số | Ý nghĩa |
|------|---------|---------|
| `ffmpeg_capture.py` | `open_timeout_sec`, `read_timeout_sec` | Timeout mở/đọc stream |
| `ffmpeg_capture.py` | `reconnect_delay_sec`, `reconnect_delay_max_sec` | Backoff khi reconnect |
| `ffmpeg_capture.py` | `_stall_timeout_sec` | Restart khi stream “đơ” |

Camera hay rớt: tăng nhẹ `read_timeout_sec` / `stall_timeout_sec` (8–10s), giữ `rtsp_transport=tcp`.

### Test nhiều camera (tùy chọn)

```python
# Trong script test hoặc REPL sau khi tạo MainWindow
from scale_test_helper import log_multi_camera_status
from Logging import Logger

logger = Logger("SCALE-TEST")
# Chạy ở thread phụ để không block UI
import threading
threading.Thread(
    target=log_multi_camera_status,
    args=(window, logger),
    kwargs={"interval_sec": 5, "duration_sec": 120},
    daemon=True,
).start()
```

---

## Cài đặt FFmpeg

Ứng dụng **tự dùng FFmpegCapture** khi `camera_src` là RTSP và lệnh `ffmpeg` có trong PATH. Không có FFmpeg → fallback OpenCV (kém ổn định hơn với RTSP dài hạn).

### Cách 1: winget (nhanh)

```powershell
winget install --id Gyan.FFmpeg
```

Đóng và mở lại terminal, kiểm tra:

```powershell
ffmpeg -version
```

### Cách 2: Tải thủ công

1. Tải bản Windows từ [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/) — dùng bản **full** (có CUDA/NVDEC), không dùng *essentials* nếu cần `h264_cuvid`.
2. Giải nén, ví dụ `C:\ffmpeg\bin\ffmpeg.exe`.
3. Thêm `C:\ffmpeg\bin` vào **Environment Variables → Path**.
4. Mở terminal mới và chạy `ffmpeg -version`.

### Kiểm tra RTSP

```powershell
ffmpeg -rtsp_transport tcp -i "rtsp://user:pass@ip/..." -t 5 -f null -
```

---

## Log & ảnh cảnh báo

- **Log ứng dụng:** qua `Logging.py` (logger tên `CCTV` trong `Program.py`).
- **File log hằng ngày:** `Log_Vision/Log_View/YYYY_MM_DD.log` (ví dụ `2026_05_16.log`).
- **Ảnh vi phạm:** lưu tại `./LastDetectionWarning/` với tên `{camera_slug}_{timestamp}_warning.jpg`.
- **Panel phải:** đọc thư mục này theo `config_data_log.json`.

### Cách mở log nhanh

```powershell
# Xem log hôm nay (PowerShell)
Get-Content "Log_Vision\Log_View\$(Get-Date -Format 'yyyy_MM_dd').log" -Wait -Tail 50

# Chỉ dòng ingest / YOLO
Select-String -Path "Log_Vision\Log_View\*.log" -Pattern "\[INGEST\]|\[YOLO\]"
```

---

## Chẩn đoán ingest / NVDEC / RAW (bảng log)

Khi mở app, hệ thống tự ghi **một lần** kiểm tra FFmpeg/GPU, sau đó **mỗi camera RTSP** ghi pipeline riêng.

### Bước 1 — Log lúc khởi động app (toàn hệ thống)

| Log bạn thấy | Ý nghĩa | Trạng thái |
|--------------|---------|------------|
| `[INGEST] ffmpeg: YES` | FFmpeg có trong PATH | Cần có cho RTSP ổn định |
| `[INGEST] ffprobe: YES` | ffprobe có trong PATH | Cần cho **raw BGR** (khuyến nghị) |
| `[INGEST] cuda_hwaccel: YES` | FFmpeg build hỗ trợ CUDA | Bước đầu để NVDEC |
| `[INGEST] h264_cuvid: YES` | NVDEC H.264 | Tối ưu cho camera CCTV phổ biến |
| `[INGEST] hevc_cuvid: YES` | NVDEC H.265/HEVC | Camera H.265 |
| `[INGEST] scale_cuda: YES` | Resize trên GPU | Giảm CPU khi scale 640px |
| `[INGEST] OK NVDEC-ready (...)` | Đủ điều kiện NVDEC H.264 | **Tốt nhất** |
| `[YOLO] OK CUDA NVIDIA GeForce RTX 4060 (...)` | PyTorch dùng GPU inference | **Tốt** |
| `[YOLO] WARN CPU only — PyTorch không thấy CUDA` | YOLO chạy CPU | Cài `torch` bản CUDA |

| Log lỗi lúc khởi động | Nguyên nhân | Cách sửa |
|----------------------|-------------|----------|
| `[INGEST] ffmpeg: NO` | Chưa cài / chưa thêm PATH | Cài FFmpeg, mở lại terminal |
| `[INGEST] WARN ffmpeg missing in PATH` | RTSP sẽ fallback OpenCV | Cài FFmpeg |
| `[INGEST] ffprobe: NO` | Thiếu ffprobe | Cài cùng gói FFmpeg |
| `[INGEST] WARN no CUDA in ffmpeg` | Bản **essentials** hoặc build không CUDA | Dùng bản **full** từ gyan.dev |
| `[INGEST] WARN cuda yes but no h264_cuvid` | CUDA có nhưng thiếu decoder NVDEC | Đổi sang FFmpeg full build |
| `[YOLO] WARN CPU only` | PyTorch CPU-only | `pip install torch --index-url .../cu121` |

### Bước 2 — Log mỗi camera RTSP (sau khi connect)

| Log bạn thấy | Pipeline thực tế | Đánh giá |
|--------------|------------------|----------|
| `[INGEST] OK \| raw 640x360\|decode=nvdec (h264_cuvid)\|scale_cuda \| native 1920x1080 codec=h264` | Raw BGR + NVDEC + scale GPU | **Chuẩn production** |
| `[INGEST] OK \| raw 640x360\|decode=nvdec (hevc_cuvid)\|scale_cuda \| ...` | Tương tự, camera H.265 | **Tốt** |
| `[INGEST] OK \| raw 640x360\|decode=nvdec (h264_cuvid)\|scale_cpu \| ...` | NVDEC OK, scale CPU | **Khá tốt** (vẫn bỏ MJPEG) |
| `[INGEST] OK \| raw 640x360\|decode=cuda\|scale_cpu \| ...` | Decode GPU generic, không cuvid | **Trung bình** |
| `[INGEST] OK \| raw 640x360\|decode=cpu\|scale_cpu \| ...` | Raw BGR nhưng decode CPU | **Chấp nhận** (vẫn nhẹ hơn MJPEG cũ) |
| `[INGEST] WARN MJPEG fallback mode` | Quay lại pipeline MJPEG cũ | **CPU cao** — sửa ffprobe/RTSP |
| `[INGEST] WARN probe failed, MJPEG fallback` | ffprobe không đọc được stream | Kiểm tra URL, firewall, credential |
| Không có dòng `[INGEST]` | `camera_src` không phải RTSP | File/USB dùng OpenCV (bình thường) |

### Bước 3 — Log runtime (trong lúc chạy 24/7)

| Log | Ý nghĩa | Cần lo không? |
|-----|---------|----------------|
| `[INGEST] start raw 640x360 \| raw 640x360\|decode=nvdec...` | FFmpeg subprocess vừa start/restart | Bình thường sau reconnect |
| `[INGEST] WARN reconnect: ...` | Mất frame tạm thời, đang backoff | Thỉnh thoảng OK |
| `[INGEST] WARN watchdog stalled stream` | Stream “đơ” không có frame mới | Kiểm tra camera/mạng |
| `[INGEST] WARN CUDA/NVDEC disabled after failure` | NVDEC lỗi, đã chuyển CPU decode | Xem driver NVIDIA / FFmpeg build |
| `[INGEST] WARN raw unstable (...), switching to MJPEG fallback` | Raw pipe lỗi liên tục | Hiếm — kiểm tra codec/resolution |
| `[INGEST] ERROR ffmpeg not found` | PATH mất khi app đang chạy | Khởi động lại app sau khi sửa PATH |
| `[YOLO] queue overloaded, dropping frame` | GPU không kịp inference | Giảm `_target_fps` hoặc số camera |
| UI **LIVE** (chấm xanh) | Có frame mới | OK |
| UI **NO SIGNAL** (vàng) | Không frame > 30s | RTSP/FFmpeg/network |

### Ma trận nhanh: bạn đang ở mức nào?

| raw | decode | scale | Kết luận |
|:---:|:------:|:-----:|----------|
| ✓ | nvdec | scale_cuda | Tối ưu đầy đủ (RTX 4060 đúng hướng) |
| ✓ | nvdec | scale_cpu | Tốt (chỉ resize còn trên CPU) |
| ✓ | cpu | scale_cpu | Khá (đã bỏ MJPEG double-decode) |
| ✗ | * | * | MJPEG fallback — cần sửa ffprobe/RTSP |
| — | — | — | Không RTSP hoặc không dùng FFmpegCapture |

### Lệnh kiểm tra trước khi chạy app (PowerShell)

```powershell
ffmpeg -version
ffmpeg -hide_banner -hwaccels
ffmpeg -hide_banner -decoders | findstr cuvid
ffprobe -version
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

Kết quả mong muốn trên RTX 4060: `cuda` trong hwaccels, có `h264_cuvid`, `torch.cuda` = `True`.

---

## Xử lý sự cố

| Triệu chứng | Nguyên nhân thường gặp | Hướng xử lý |
|-------------|------------------------|-------------|
| NO SIGNAL liên tục | RTSP sai URL, mạng, hoặc chưa cài FFmpeg | Test bằng lệnh `ffmpeg` ở trên; kiểm tra firewall |
| FPS thấp / lag | Quá nhiều camera hoặc `_target_fps` cao | Giảm `_target_fps`, `img_size`, hoặc số camera bật |
| GPU đầy / OOM | Model quá lớn hoặc batch cao | Dùng model nhỏ hơn (`yolo11n.pt`); giảm `_batch_size` |
| Cảnh báo sai / không báo | `yolo_rate` hoặc class index sai model | Chỉnh `yolo_rate`; kiểm tra cls `5/8/9` trong `yolo_engine.py` |
| Queue overloaded (log) | GPU không kịp | Giảm FPS/camera hoặc tăng `yolo_rate` |
| Widget camera phình to | QLabel scale theo pixmap | Đã xử lý bằng `setScaledContents(False)` — nếu tái hiện, kiểm tra `resizeEvent` |

---

## Huấn luyện model (tùy chọn)

File `yolo_training.py` và `data.yaml` dùng để train/fine-tune YOLO. Sau khi có weight mới, cập nhật `yolo_model_path` trong config và đảm bảo class index trong `yolo_engine.py` khớp với model.

---

## Giấy phép & liên hệ

Dự án nội bộ EHS CCTV. Chỉnh sửa và triển khai theo quy trình của đơn vị.
