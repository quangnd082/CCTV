### Giải thích ngắn gọn các thay đổi đề xuất
"""
- **Giảm độ trễ đọc frame (capture latency)**
  - Đổi `self.q = queue.Queue()` thành `Queue(maxsize=1)` và xóa frame cũ trước khi put: luôn giữ “frame mới nhất”, tránh backlog → hình hiển thị mượt hơn.
  - Đặt thread đọc camera là `daemon=True` và join nhẹ khi `release()`: thoát app sạch sẽ, không treo vì thread nền.

- **Tối ưu suy luận YOLO (inference)**
  - Thêm `torch` và tự chọn `device`: dùng GPU nếu có (`cuda`) hoặc CPU nếu không.
  - Bật `half` precision khi có CUDA: giảm băng thông/compute → inference nhanh hơn.
  - Gom các tham số predict vào `self.predict_params` và tái sử dụng mỗi frame: giảm overhead khởi tạo và log `verbose`.
  - Dùng `results[0]` (1 ảnh → 1 kết quả) thay vì lặp qua `results`: ít vòng lặp không cần thiết.

- **Giảm nghẽn I/O khi lưu cảnh báo**
  - Thay vì encode và ghi file ngay trong loop, chuyển sang một thread nền: `threading.Thread(..., daemon=True).start()` với `frame.copy()` để an toàn bộ nhớ.
  - Thêm throttle 2 giây (`self._last_warn_ts`) cho việc lưu file warning: không spam I/O liên tục → vòng lặp detect mượt hơn.

- **Giữ UI mượt**
  - Vẫn chạy detect trong `QThread` và chỉ emit `QImage` về UI thread → không block UI.
  - Không đổi logic ROI, màu, vẽ khung; chỉ tinh gọn đường đi dữ liệu và I/O.

- **Tùy chọn tăng mượt thêm (bật khi cần)**
  - Giảm `img_size` (ví dụ 512/416) khi tạo `CameraWidget`.
  - Máy chỉ CPU: xử lý cách frame (skip mỗi 1 frame).
  - Downscale khung 1080p trước khi detect (`cv2.resize` với `INTER_AREA`).

- **Tương thích và an toàn**
  - `half` chỉ bật khi `torch.cuda.is_available()` → không lỗi trên CPU.
  - Dọn tài nguyên camera + thread gọn gàng trong `release()` và `stop()`.

- **Tác động chính**
  - Ít trễ hơn giữa camera → màn hình.
  - Inference nhanh/nhẹ hơn khi có GPU.
  - Vòng lặp detect ổn định vì không bị chặn bởi việc ghi file.

  - Mục tiêu FPS bao nhiêu và máy bạn có GPU nào? Mình sẽ set `img_size`, tỉ lệ downscale và nhịp skip frame phù hợp.

- Đã tối ưu capture (queue 1 phần tử + daemon thread), inference (GPU/half + tham số tái sử dụng), và I/O (encode ảnh nền + throttle).
- Kết quả: giảm latency, giảm giật khung và ổn định UI khi cảnh báo liên tục.
"""
import queue
import threading
import time

from PIL import Image
import io

import cv2
import torch
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter, QPen
from PyQt5.QtCore import QTimer, QSize, Qt, QThread, pyqtSignal
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from datetime import datetime
import base64
from unidecode import unidecode

from WidgetCustom import *
from queue import Queue
import gc
from display_image import *

import os, re
from pathlib import Path

INVALID_CHARS = r'[^A-Za-z0-9_.-]'




def safe_name(s: str) -> str:
    """Biến chuỗi thành tên file an toàn cho Windows."""
    s = os.path.basename(str(s))  # nếu là path, chỉ lấy tên cuối
    s = s.replace(':', '_')  # bỏ dấu :
    s = re.sub(INVALID_CHARS, '_', s)  # thay kí tự lạ
    return s[:80]


class VideoCapture:

    def __init__(self, name):
        self.source = name
        self.is_file = isinstance(name, str) and os.path.isfile(name)
        self.cap = cv2.VideoCapture(name)
        if not self.is_file:
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        self.q = queue.Queue(maxsize=1)
        self._t = threading.Thread(target=self._reader, name=f"VideoReader-{self.source}", daemon=True)
        self._t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("FAIL READ FRAME")
                time.sleep(0.1)
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        try:
            return self.q.get(timeout=0.5)
        except queue.Empty:
            return None

    def release(self):
        try:
            self.cap.release()
        finally:
            if hasattr(self, '_t') and self._t.is_alive():
                try:
                    self._t.join(timeout=0.2)
                except Exception:
                    pass


class CameraThread(QThread):
    frame_captured = pyqtSignal(QImage)
    warning_changed = pyqtSignal(bool)

    def __init__(self, camera_id, model, yolo_rate, img_size, roi_check, classes, colors,
                 enable_flags, camera_name=None):
        super().__init__()
        self.camera_id = camera_id
        self.model = model
        self.yolo_rate = yolo_rate
        self.img_size = img_size
        self.roi_check = roi_check or [-1, 9999, -1, 9999]
        self.classes = classes
        self.colors = colors
        self.enable_flags = enable_flags
        self.running = True
        self.video_capture = VideoCapture(camera_id)

        self.camera_name = camera_name or str(self.camera_id)
        self.camera_slug = safe_name(unidecode(self.camera_name))
        self.save_dir = Path("./LastDetectionWarning")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Inference config
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.predict_params = {
            'imgsz': self.img_size,
            'conf': float(self.yolo_rate),
            'device': self.device,
            'verbose': False,
            'agnostic_nms': True
        }
        if torch.cuda.is_available():
            self.predict_params['half'] = True
        self._last_warn_ts = 0.0
        self._last_warning_state = False

    def run(self):
        x1, x2, y1, y2 = self.roi_check
        while self.running:
            frame = self.video_capture.read()
            if frame is None:
                time.sleep(0.01)
                continue
            results = self.model.predict(frame, **self.predict_params)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotator = Annotator(cv2image, line_width=1, font_size=16)
            is_warning = False
            r = results[0]
            boxes = r.boxes
            for box in boxes:
                try:
                    if float(box.conf) <= float(self.yolo_rate):
                        continue
                except Exception:
                    pass
                b = box.xyxy[0]
                try:
                    bx1, by1, bx2, by2 = [float(v) for v in b]
                except Exception:
                    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                if bx1 < x1 or by1 < y1 or bx2 > x2 or by2 > y2:
                    continue
                ci = int(box.cls)
                helmet_ok = (ci == 5 and self.enable_flags.get('helmet', False))
                fell_ok = (ci == 6 and self.enable_flags.get('fell', False))
                jacket_ok = (ci == 7 and self.enable_flags.get('jacket', False))


                # an toàn label & màu
                label = str(self.classes[ci]) if 0 <= ci < len(self.classes) and self.classes[ci] else str(ci)
                color_warn = tuple(self.colors[ci]) if 0 <= ci < len(self.colors) else (255, 0, 0)

                if helmet_ok or fell_ok or jacket_ok :
                    is_warning = True
                    annotator.box_label([bx1, by1, bx2, by2], label, color=color_warn)
                else:
                    annotator.box_label([bx1, by1, bx2, by2], label, color=(0, 255, 0))
            img = annotator.result()
            h, w, _ = img.shape
            if is_warning:
                annotator.box_label([5, 5, w - 5, h - 5], "", color=(255, 0, 0))
                img = annotator.result()
                now_ts = time.time()
                if now_ts - self._last_warn_ts > 10.0:
                    print(f"Time save Image = {now_ts - self._last_warn_ts}")
                    try:
                        ts = datetime.now().strftime("%Y%m%d%H%M%S")
                        file_path = self.save_dir / f"{self.camera_slug}_{ts}_warning.jpg"
                        threading.Thread(
                            target=self.save_image,
                            args=(str(file_path), img.copy()),
                            daemon=True
                        ).start()
                        self._last_warn_ts = now_ts
                    except Exception as e:
                        print("save image start error:", e)
            # báo trạng thái warning ra UI nếu có thay đổi
            if is_warning != self._last_warning_state:
                self._last_warning_state = is_warning
                self.warning_changed.emit(is_warning)
            q_image = QImage(img.data, w, h, w * 3, QImage.Format_RGB888).copy()
            self.frame_captured.emit(q_image)

    def save_image(self, file_path, img_rgb):
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, img_bgr)
        except Exception as e:
            print("save image error:", e)

    def stop(self):
        self.running = False
        self.video_capture.release()
        self.wait()


class CameraWidget(QWidget):
    def __init__(self, camera_name="None", camera_src=0, img_size=640, yolo_model_path="", yolo_rate=0.5,
                 classes=["", "Helmet", "Fall", "No Helmet"], roi_check=[],
                 colors=[(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 0)], is_fell_check=True, is_fire_check=False,
                 is_smoke_check=False,
                 is_heltmet_check=False, is_jacket_check=False, timer_delay=10, parent=None):
        super().__init__(parent)
        self.frame_count = None
        self.camera_name = camera_name
        self.img_size = img_size
        self.yolo_model_path = yolo_model_path
        self.yolo_rate = yolo_rate
        self.is_fell_check = is_fell_check
        self.is_helmet_check = is_heltmet_check
        self.is_jacket_check = is_jacket_check
        self.is_fire_check = is_fire_check
        self.is_smoke_check = is_smoke_check
        self.roi_check = roi_check

        self.classes = classes
        self.colors = colors

        self.timer_delay = timer_delay
        self.is_warning = False
        self.initUI(camera_name)
        self.model = YOLO(yolo_model_path)
        # Fuse + move to device + half (nếu CUDA)
        try:
            self.model.fuse()
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'fuse'):
                self.model.model.fuse()
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                self.model.to('cuda')
                self.model.model.half()
            except Exception:
                pass

        # Start CameraThread (detect+annotate in worker thread, UI receives QImage via signal)
        i = int(camera_src) if isinstance(camera_src, int) else camera_src
        print(i)
        self.enable_flags = {
            'fell': self.is_fell_check,
            'helmet': self.is_helmet_check,
            'jacket': self.is_jacket_check,
            'fire': self.is_fire_check,
            'smoke': self.is_smoke_check
        }
        self.camera_thread = CameraThread(
            camera_id=i,
            model=self.model,
            yolo_rate=self.yolo_rate,
            img_size=self.img_size,
            roi_check=self.roi_check,
            classes=self.classes,
            colors=self.colors,
            enable_flags=self.enable_flags,
            camera_name=self.camera_name
        )
        self.camera_thread.frame_captured.connect(self._on_frame)
        self.camera_thread.warning_changed.connect(self._on_warning_changed)
        self.camera_thread.start()



    def initUI(self, camera_name="None"):

        # Grid tổng
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)
        # Label hiển thị ảnh
        self.image_label = QLabel(self)

        # self.image_label.setStyleSheet("border :5px solid green;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.layout.addWidget(self.image_label, 0, 0, 3, 3)  # Đặt label vào ô (0, 1)


        # Label hiển thị tên camera
        self.camera_name_label = QLabel(self)
        self.camera_name_label.setStyleSheet(
            "font-size: 25px; font-weight: bold; padding-left : 5px;padding-right: 0px;padding-bottom: 180px;margin-top: 10px;padding-top: 0px;  color : white;")
        self.camera_name_label.setText(camera_name)
        self.layout.addWidget(self.camera_name_label, 0, 0, 1, 0)

        # Layout hiển tḥ check box
        self.checkbox_layout = QVBoxLayout(self)

        # Checkbox Helmet check
        self.checkbox_helmet = QCheckBox("Helmet")
        self.checkbox_helmet.setStyleSheet("font-size: 16px;padding-right: 10px; font-weight: bold;  color : white;")
        self.checkbox_helmet.setChecked(self.is_helmet_check)
        self.checkbox_helmet.stateChanged.connect(self.onHelmetStateChange)

        # Checkbox Fell check
        self.checkbox_fell = QCheckBox("Fell")
        self.checkbox_fell.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 0px; margin-top: 0px;color: white;")
        self.checkbox_fell.setChecked(self.is_fell_check)
        self.checkbox_fell.stateChanged.connect(self.onFellStateChange)

        # Checkbox Jacket check
        self.checkbox_jacket = QCheckBox("Jacket")
        self.checkbox_jacket.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 0px; margin-top: 0px;color: white;")
        self.checkbox_jacket.setChecked(self.is_jacket_check)
        self.checkbox_jacket.stateChanged.connect(self.onJacketStateChange)

        self.checkbox_fire = QCheckBox("Fire")
        self.checkbox_fire.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 0px; margin-top: 0px;color: white;")
        self.checkbox_fire.setChecked(self.is_fire_check)
        self.checkbox_fire.stateChanged.connect(self.onFireStateChange)

        # Checkbox Smoke check
        self.checkbox_smoke = QCheckBox("Smoke")
        self.checkbox_smoke.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 0px; margin-top: 0px;color: white;")
        self.checkbox_smoke.setChecked(self.is_smoke_check)
        self.checkbox_smoke.stateChanged.connect(self.onSmokeStateChange)

        # self.checkbox_layout.addWidget(self.checkbox_fire)
        # self.checkbox_layout.addWidget(self.checkbox_smoke)
        self.checkbox_layout.addWidget(self.checkbox_fell)
        self.checkbox_layout.addWidget(self.checkbox_helmet)
        self.checkbox_layout.addWidget(self.checkbox_jacket)

        # self.checkbox_layout.setStyleSheet("padding-left: 5px; padding-right: 3px;padding-top: 0px;")

        # Label hiển thị thời gian
        self.label_date_time = QLabel()
        self.label_date_time.setStyleSheet("font-size: 15px; font-weight: bold; padding : 5px;  color : white;")
        self.layout.addWidget(self.label_date_time, 2, 0)

        # Label hiển thị WARNING
        self.label_warning = QLabel()
        self.label_warning.setText("WARNING")
        self.label_warning.setAlignment(Qt.AlignHCenter)
        self.label_warning.setAlignment(Qt.AlignVCenter)

        self.label_warning.setVisible(False)
        self.label_warning.setContentsMargins(0, 0, 0, 0)
        self.label_warning.setStyleSheet(
            "font-size: 40px; font-weight: bold; color : red;")
        self.layout.addWidget(self.label_warning, 1, 1,1,1, Qt.AlignCenter)

        # Set chiều rộng, cao cho các grid
        self.layout.setRowMinimumHeight(2, 50)
        self.layout.setRowMinimumHeight(0, 50)

        self.layout.addLayout(self.checkbox_layout, 0, 2)  # Đặt checkbox vào ô (0, 0)
        # Đặt layout chính cho widget

        self.image_folder_path = unidecode(self.camera_name).replace(" ", "")

        self.frame_count = 0

    def _on_frame(self, q_image: QImage):
        # giữ aspect & smoothing, nhưng chỉ nên dùng nếu cần chất lượng cao
        # pm = QPixmap.fromImage(q_image)
        # pm = pm.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        # self.image_label.setPixmap(pm)
        self.image_label.setPixmap(QPixmap.fromImage(q_image).scaled(self.image_label.size()))

    def _on_warning_changed(self, flag: bool):
        self.is_warning = flag
        if self.is_warning:
            self.label_warning.setVisible(not self.label_warning.isVisible())
        else:
            self.label_warning.setVisible(False)

    def onHelmetStateChange(self, state):
        if state == 2:  # Qt.Checked = 2
            self.is_helmet_check = True
        else:
            self.is_helmet_check = False
        if hasattr(self, 'enable_flags'):
            self.enable_flags['helmet'] = self.is_helmet_check

    def onFellStateChange(self, state):
        if state == 2:  # Qt.Checked = 2
            self.is_fell_check = True
        else:
            self.is_fell_check = False
        if hasattr(self, 'enable_flags'):
            self.enable_flags['fell'] = self.is_fell_check

    def onJacketStateChange(self, state):
        if state == 2:  # Qt.Checked = 2
            self.is_jacket_check = True
        else:
            self.is_jacket_check = False
        if hasattr(self, 'enable_flags'):
            self.enable_flags['jacket'] = self.is_jacket_check

    def onFireStateChange(self, state):
        if state == 2:  # Qt.Checked = 2
            self.is_fire_check = True
        else:
            self.is_fire_check = False
        if hasattr(self, 'enable_flags'):
            self.enable_flags['fire'] = self.is_fire_check

    def onSmokeStateChange(self, state):
        if state == 2:  # Qt.Checked = 2
            self.is_smoke_check = True
        else:
            self.is_smoke_check = False
        if hasattr(self, 'enable_flags'):
            self.enable_flags['smoke'] = self.is_smoke_check

    def update_date_time(self):
        self.label_date_time.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        if self.is_warning:
            self.label_warning.setVisible(not self.label_warning.isVisible())
        else:
            self.label_warning.setVisible(False)

    def closeEvent(self, event):
        try:
            if hasattr(self, 'camera_thread'):
                self.camera_thread.stop()
        except Exception:
            pass
        super().closeEvent(event)

    def resizeEvent(self, e):
        if hasattr(self, 'camera_thread'):
            self.camera_thread.display_size = (self.image_label.width(), self.image_label.height())
        super().resizeEvent(e)



