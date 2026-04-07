# CCTV

## Cài đặt FFmpeg (khuyến nghị cho RTSP ổn định 24/7)

Ứng dụng sẽ **tự động dùng FFmpegCapture cho RTSP** nếu máy có `ffmpeg` trong PATH. Nếu không có, ứng dụng sẽ fallback sang OpenCV capture.

### Cách 1: Cài nhanh bằng winget

Mở PowerShell (Admin) và chạy:

```powershell
winget install --id Gyan.FFmpeg
```

Đóng/mở lại terminal, kiểm tra:

```powershell
ffmpeg -version
```

### Cách 2: Tải bản release và thêm vào PATH

1. Tải FFmpeg Windows build từ `https://www.gyan.dev/ffmpeg/builds/` (bản “essentials” là đủ).
2. Giải nén vào ví dụ `C:\ffmpeg\` để có `C:\ffmpeg\bin\ffmpeg.exe`.
3. Thêm `C:\ffmpeg\bin` vào **Environment Variables → Path**.
4. Mở terminal mới và kiểm tra:

```powershell
ffmpeg -version
```

### Test nhanh RTSP bằng FFmpeg (tuỳ chọn)

```powershell
ffmpeg -rtsp_transport tcp -i "rtsp://user:pass@ip/..." -t 5 -f null -
```

## Cấu hình camera (`config_data.json`)

File cấu hình chính: `config_data.json`.

- **Bật/tắt camera**: dùng `enable_flags.use_camera`
  - `1`: chạy camera (hiển thị + capture + inference)
  - `0`: bỏ qua camera (không tạo widget)
- **Bật/tắt rule cảnh báo**: trong `enable_flags` (ví dụ `fell/helmet/jacket/fire/smoke`)

Ví dụ:

```json
{
  "camera_name": "CCTV 01",
  "camera_src": "rtsp://user:pass@ip/...",
  "img_size": 640,
  "yolo_model_path": "yolo11n.pt",
  "yolo_rate": 0.5,
  "enable_flags": {
    "use_camera": 1,
    "fell": 1,
    "helmet": 1,
    "jacket": 0,
    "fire": 0,
    "smoke": 0
  }
}
```

## Luồng xử lý (từ UI → Capture → YOLO → UI)

### 1) Khởi động ứng dụng

- Entry point: `Program.py`
- `MainWindow` load `config_data.json`, lọc camera theo `enable_flags.use_camera`, sau đó tự động tính layout grid theo số camera đang bật.

### 2) Capture cho từng camera

- Mỗi `CameraWidget` tạo `VideoCapture(camera_src)` trong `CameraWidget.py`.
- Nếu `camera_src` là RTSP và có `ffmpeg` trong PATH:
  - dùng `FFmpegCapture` (`ffmpeg_capture.py`) để đọc RTSP bền hơn 24/7 (auto-restart + exponential backoff).
- Nếu không:
  - dùng OpenCV `cv2.VideoCapture` + reconnect/backoff.
- Capture giữ queue `maxsize=1` để luôn lấy frame mới nhất (giảm latency, tránh backlog).

### 3) YOLO inference dùng chung (batch + throttle)

- `YoloEngine` (`yolo_engine.py`) là engine dùng chung:
  - cache model theo `yolo_model_path` (không load lặp theo camera)
  - gom batch (tối đa 4 frame/lần) nếu các job tương thích (`model_path/img_size/yolo_rate`)
- Mỗi `CameraWidget` gửi frame vào engine theo nhịp:
  - có throttle per-camera (`_target_fps`) + cờ `_infer_in_flight` để không dồn queue.

### 4) Trả kết quả về UI

- `YoloEngine` emit signal `result_ready(camera_name, q_image, is_warning, meta)`.
- `CameraWidget` nhận kết quả đúng camera, hiển thị lên `QLabel` và bật/tắt overlay cảnh báo theo trạng thái warning.

## Tuning hiệu năng (tăng/giảm tốc độ xử lý)

Phần này liệt kê các thông số quan trọng để bạn “chốt cấu hình mượt” cho 4–6 camera RTSP.

### 1) Giảm/tăng tần số YOLO (ảnh hưởng trực tiếp GPU)

- **File**: `CameraWidget.py`
- **Biến**: `self._target_fps`
  - Ví dụ hiện tại mặc định ưu tiên ổn định 24/7:
    - `self._target_fps = 6.0`
  - Gợi ý:
    - 4 camera: `8–10` FPS/cam (nếu GPU còn dư)
    - 6 camera: `6–8` FPS/cam (ưu tiên bền)

Tăng `self._target_fps` → **mượt hơn** nhưng GPU tải cao hơn. Giảm → **bền hơn** và ít drop.

### 2) Kích thước input YOLO (ảnh hưởng GPU + chất lượng)

- **File**: `config_data.json`
- **Field**: `img_size`
  - Thường dùng `640` là cân bằng.
  - Nếu muốn nhẹ hơn: `512`.

Giảm `img_size` → **nhanh hơn** nhưng độ chính xác/độ chi tiết có thể giảm.

### 3) Ngưỡng confidence (ảnh hưởng số box + tốc độ vẽ/logic)

- **File**: `config_data.json`
- **Field**: `yolo_rate`
  - Tăng (vd `0.6–0.7`) → ít box hơn, đỡ nhiễu, xử lý nhẹ hơn.
  - Giảm (vd `0.3–0.4`) → nhiều box hơn, nặng hơn và dễ false-positive.

### 4) Batch inference (tận dụng GPU tốt hơn)

- **File**: `yolo_engine.py`
- **Biến**: `self._batch_size`
  - Mặc định: `4`
  - Gợi ý:
    - Muốn throughput tốt: giữ `4`
    - Muốn latency “đều” hơn: thử `2–3`

### 5) Queue inference (độ trễ vs khả năng chịu spike)

- **File**: `yolo_engine.py`
- **Biến**: `queue.Queue(maxsize=...)`
  - Mặc định khuyến nghị cho 4–6 cam: `16`
  - Nguyên tắc:
    - `maxsize` nhỏ → **latency thấp hơn** khi overload
    - `maxsize` lớn → chịu spike tốt hơn nhưng có thể “trễ dần”

### 6) Capture RTSP: ưu tiên ổn định 24/7

- **File**: `ffmpeg_capture.py`
  - `open_timeout_sec`, `read_timeout_sec`: timeout mở/đọc
  - `reconnect_delay_sec`, `reconnect_delay_max_sec`: backoff reconnect
  - `self._stall_timeout_sec`: watchdog restart khi stream “đơ”

Nếu bạn thấy camera hay rớt:
- tăng `read_timeout_sec` hoặc tăng `stall_timeout_sec` nhẹ (vd 8–10s)
- giữ RTSP TCP

