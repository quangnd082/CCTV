import queue
import threading
import time

from PIL import Image
import io

import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter, QPen
from PyQt5.QtCore import QTimer, QSize, Qt, QThread, pyqtSignal
from datetime import datetime
import base64
from unidecode import unidecode

from WidgetCustom import *
from queue import Queue
import gc
from display_image import *

import os
from pathlib import Path
from Logging import Logger

from yolo_engine import get_yolo_engine, safe_name


class VideoCapture:

    def __init__(self, name, logger: Logger = None):
        self.logger = logger
        self.source = name
        self.is_file = isinstance(name, str) and os.path.isfile(name)
        self.cap = None
        self.running = True

        # mở kết nối lần đầu
        self._open_capture()

        self.q = queue.Queue(maxsize=1)
        self._t = threading.Thread(target=self._reader, name=f"VideoReader-{self.source}", daemon=True)
        self._t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        """
        Vòng lặp đọc frame với cơ chế reconnect cho RTSP:
        - Nếu mất kết nối (ret=False), thử mở lại sau một khoảng delay.
        - Tránh break hẳn thread để RTSP tạm mất tín hiệu không làm crash app.
        """
        reconnect_delay = 2.0  # giây
        while self.running:
            # đảm bảo cap đang mở; nếu không, thử mở lại
            if self.cap is None or not self.cap.isOpened():
                self._open_capture()
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(reconnect_delay)
                    continue

            ret, frame = self.cap.read()
            if not ret:
                if self.logger:
                    self.logger.warning(f"FAIL READ FRAME from {self.source}, try reconnect...")
                else:
                    print(f"FAIL READ FRAME from {self.source}, try reconnect...")
                # đóng lại và chờ rồi reconnect
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                time.sleep(reconnect_delay)
                continue
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
            self.running = False
            if self.cap is not None:
                self.cap.release()
        finally:
            if hasattr(self, '_t') and self._t.is_alive():
                try:
                    self._t.join(timeout=0.2)
                except Exception:
                    pass

    def _open_capture(self):
        """Mở hoặc mở lại VideoCapture với cấu hình phù hợp (RTSP/USB/file)."""
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.is_file and self.cap is not None and self.cap.isOpened():
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error opening capture {self.source}: {e}")
            else:
                print(f"Error opening capture {self.source}: {e}")
            self.cap = None


class CameraThread(QThread):
    frame_captured = pyqtSignal(QImage)
    warning_changed = pyqtSignal(bool)

    def __init__(self, camera_id, model, yolo_rate, img_size, roi_check, classes, colors,
                 enable_flags, camera_name=None, logger: Logger = None):
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
        self.logger = logger
        self.video_capture = VideoCapture(camera_id, logger=self.logger)

        self.camera_name = camera_name or str(self.camera_id)
        self.camera_slug = safe_name(unidecode(self.camera_name))
        self.save_dir = Path("./LastDetectionWarning")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Inference config
        # giữ cấu trúc cũ nhưng không còn dùng trực tiếp trong kiến trúc mới
        self.device = 0
        self.predict_params = {
            'imgsz': self.img_size,
            'conf': float(self.yolo_rate),
            'device': self.device,
            'verbose': False,
            'agnostic_nms': True
        }
        self._last_warn_ts = 0.0
        self._last_warning_state = False

    def run(self):
        x1, x2, y1, y2 = self.roi_check
        while self.running:
            frame = self.video_capture.read()
            if frame is None:
                time.sleep(0.01)
                continue
            # logic cũ hiện không còn được sử dụng trong kiến trúc mới
            results = self.model.predict(frame, **self.predict_params)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotator = None
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
                fell_ok = (ci == 9 and self.enable_flags.get('fell', False))
                jacket_ok = (ci == 8 and self.enable_flags.get('jacket', False))

                # an toàn label & màu
                # ưu tiên tên trong config; nếu trống hoặc thiếu thì fallback sang model.names
                try:
                    name_from_cfg = self.classes[ci] if 0 <= ci < len(self.classes) else None
                except Exception:
                    name_from_cfg = None
                name_from_model = None
                try:
                    if hasattr(self.model, 'names') and self.model.names is not None:
                        # ultralytics .names có thể là dict hoặc list
                        if isinstance(self.model.names, dict):
                            name_from_model = self.model.names.get(ci)
                        elif 0 <= ci < len(self.model.names):
                            name_from_model = self.model.names[ci]
                except Exception:
                    pass
                obj_name = name_from_cfg if (isinstance(name_from_cfg, str) and len(name_from_cfg) > 0) else (
                    name_from_model if name_from_model else str(ci))
                try:
                    conf_val = float(box.conf)
                except Exception:
                    conf_val = 0.0
                label = f"{obj_name} {conf_val:.2f}"
                color_warn = tuple(self.colors[ci]) if 0 <= ci < len(self.colors) else (255, 0, 0)

                if helmet_ok or fell_ok or jacket_ok:
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
                if now_ts - self._last_warn_ts > 5.0:

                    try:
                        ts = datetime.now().strftime("%Y%m%d%H%M%S")
                        file_path = self.save_dir / f"{self.camera_slug}_{ts}_warning.jpg"
                        threading.Thread(
                            target=self.save_image,
                            args=(str(file_path), img.copy()),
                            daemon=True
                        ).start()
                        self._last_warn_ts = time.time()
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"save image start error: {e}")
                        else:
                            print(f"save image start error: {e}")
            # báo trạng thái warning ra UI nếu có thay đổi
            if is_warning != self._last_warning_state:
                self._last_warning_state = is_warning
                if is_warning:
                    if self.logger:
                        self.logger.warning(f"Violation detected")
                    else:
                        print(f"Violation detected")
                self.warning_changed.emit(is_warning)
            q_image = QImage(img.data, w, h, w * 3, QImage.Format_RGB888).copy()
            self.frame_captured.emit(q_image)

    def save_image(self, file_path, img_rgb):
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, img_bgr)
        except Exception as e:
            if self.logger:
                self.logger.error(f"save image error: {e}")
            else:
                print(f"save image error: {e}")

    def stop(self):
        self.running = False
        self.video_capture.release()
        self.wait()


class CameraWidget(QWidget):
    def __init__(self, camera_name="None", camera_src=0, img_size=640, yolo_model_path="", yolo_rate=0.5,
                 classes=["", "Helmet", "Fall", "No Helmet"], roi_check=[],
                 colors=[(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 0)], is_fell_check=True, is_fire_check=False,
                 is_smoke_check=False,
                 is_heltmet_check=False, is_jacket_check=False, timer_delay=10, parent=None, logger: Logger = None):
        super().__init__(parent)
        self.frame_count = None
        self.camera_name = camera_name
        self.camera_src = camera_src
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
        self.logger = logger

        # Engine YOLO dùng chung
        self.engine = get_yolo_engine()

        # Video capture riêng cho từng camera (thread nhẹ đọc frame)
        self.video_capture = VideoCapture(camera_src, logger=self.logger)

        # Cờ và tham số điều phối FPS gửi vào YOLO
        self._infer_in_flight = False
        self._target_fps = 8.0  # tối đa 8 inference/giây cho mỗi camera
        self._infer_interval = 1.0 / self._target_fps
        self._last_sent_ts = 0.0

        # Log camera ID
        i = int(camera_src) if isinstance(camera_src, int) else camera_src
        if self.logger:
            self.logger.info(f"Camera ID = {i}")
        else:
            print(f"Camera ID = {i}")
        self.enable_flags = {
            'fell': self.is_fell_check,
            'helmet': self.is_helmet_check,
            'jacket': self.is_jacket_check,
            'fire': self.is_fire_check,
            'smoke': self.is_smoke_check
        }

        # Nhận kết quả inference từ YoloEngine (broadcast, mỗi widget tự lọc theo camera_name)
        self.engine.result_ready.connect(self._on_infer_result)

        # Timer để lấy frame mới và gửi request inference (đã throttle)
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self._on_frame_timer)
        # 30 ms ~ 33 FPS đọc frame; inference vẫn bị giới hạn bởi _target_fps
        self.frame_timer.start(30)

        # UI warning display throttle: show at most once every 10 minutes
        self._last_ui_warn_ts = 0.0
        self._warn_show_ms = 2000  # show label for 2 seconds when allowed
        self._warn_cooldown_sec = 600.0  # 10 minutes cooldown between shows

    def initUI(self, camera_name="None"):

        # Grid tổng
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)
        # Label hiển thị ảnh
        self.image_label = QLabel(self)

        # Style camera tile: nền tối, border subtle để UI chuyên nghiệp hơn
        self.image_label.setStyleSheet(
            "background-color: #101010; border: 1px solid #404040;"
        )
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.layout.addWidget(self.image_label, 0, 0, 3, 3)  # Đặt label vào ô (0, 1)

        # Label hiển thị tên camera
        self.camera_name_label = QLabel(self)
        self.camera_name_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; padding-left: 5px;"
            "padding-right: 0px; margin-top: 6px; padding-top: 0px; color: #f0f0f0;")
        self.camera_name_label.setText(camera_name)
        self.layout.addWidget(self.camera_name_label, 0, 0, 1, 0)

        # Layout hiển tḥ check box
        self.checkbox_layout = QVBoxLayout(self)

        # Checkbox Helmet check
        self.checkbox_helmet = QCheckBox("Helmet")
        self.checkbox_helmet.setStyleSheet(
            "font-size: 14px; padding-right: 10px; font-weight: bold; color: #e0e0e0;")
        self.checkbox_helmet.setChecked(self.is_helmet_check)
        self.checkbox_helmet.stateChanged.connect(self.onHelmetStateChange)

        # Checkbox Fell check
        self.checkbox_fell = QCheckBox("Fell")
        self.checkbox_fell.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 0px; margin-top: 0px; color: #e0e0e0;")
        self.checkbox_fell.setChecked(self.is_fell_check)
        self.checkbox_fell.stateChanged.connect(self.onFellStateChange)

        # Checkbox Jacket check
        self.checkbox_jacket = QCheckBox("Jacket")
        self.checkbox_jacket.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 0px; margin-top: 0px; color: #e0e0e0;")
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
        self.label_date_time.setStyleSheet(
            "font-size: 13px; font-weight: bold; padding: 5px; color: #c0c0c0;")
        self.layout.addWidget(self.label_date_time, 2, 0)

        # Label hiển thị WARNING
        self.label_warning = QLabel()
        self.label_warning.setText("WARNING")
        self.label_warning.setAlignment(Qt.AlignHCenter)
        self.label_warning.setAlignment(Qt.AlignVCenter)

        self.label_warning.setVisible(False)
        self.label_warning.setContentsMargins(0, 0, 0, 0)
        self.label_warning.setStyleSheet(
            "font-size: 32px; font-weight: bold; color: #ff4040;"
            "background-color: rgba(0, 0, 0, 180); padding: 8px;")
        self.layout.addWidget(self.label_warning, 1, 1, 1, 1, Qt.AlignCenter)

        # Set chiều rộng, cao cho các grid
        self.layout.setRowMinimumHeight(2, 50)
        self.layout.setRowMinimumHeight(0, 50)

        self.layout.addLayout(self.checkbox_layout, 0, 2)  # Đặt checkbox vào ô (0, 0)
        # Đặt layout chính cho widget

        self.image_folder_path = unidecode(self.camera_name).replace(" ", "")

        self.frame_count = 0

    def _on_frame_timer(self):
        """Đọc frame mới và (nếu đủ điều kiện) gửi request inference vào YoloEngine."""
        frame = self.video_capture.read()
        if frame is None:
            return

        now_ts = time.monotonic()
        if (not self._infer_in_flight) and (now_ts - self._last_sent_ts >= self._infer_interval):
            self._infer_in_flight = True
            self._last_sent_ts = now_ts
            self.engine.request_inference(
                camera_name=self.camera_name,
                frame=frame,
                model_path=self.yolo_model_path,
                img_size=self.img_size,
                yolo_rate=self.yolo_rate,
                roi_check=self.roi_check,
                classes=self.classes,
                colors=self.colors,
                enable_flags=self.enable_flags,
                logger=self.logger,
            )

    def _on_infer_result(self, camera_name: str, q_image: QImage, is_warning: bool, meta: dict):
        """Nhận kết quả inference từ YoloEngine (broadcast cho tất cả camera)."""
        if camera_name != self.camera_name:
            return

        self._infer_in_flight = False
        self.is_warning = is_warning

        # Cập nhật hình ảnh với giữ tỉ lệ và smoothing cho UI đẹp hơn
        pix = QPixmap.fromImage(q_image).scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pix)

        # Cập nhật cảnh báo trên UI (giữ cơ chế cooldown)
        self._on_warning_changed(is_warning)

    def _on_warning_changed(self, flag: bool):
        self.is_warning = flag
        if not self.is_warning:
            if hasattr(self, "label_warning") and self.label_warning:
                self.label_warning.setVisible(False)
            return
        # Only show the label if cooldown has passed
        now_ts = time.monotonic()
        if (now_ts - getattr(self, '_last_ui_warn_ts', 0.0)) >= getattr(self, '_warn_cooldown_sec', 600.0):
            self.label_warning.setVisible(True)
            self._last_ui_warn_ts = now_ts
            try:
                if hasattr(self, "label_warning") and self.label_warning:
                    self.label_warning.setVisible(True)
                    QTimer.singleShot(int(getattr(self, '_warn_show_ms', 2000)), self._hide_warning_label)
            except Exception:
                pass

    def _hide_warning_label(self):
        try:
            self.label_warning.setVisible(False)
        except Exception:
            pass

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
        # Do not blink the warning label here; visibility is controlled by _on_warning_changed cooldown

    def closeEvent(self, event):
        try:
            if hasattr(self, "frame_timer"):
                self.frame_timer.stop()
            if hasattr(self, "video_capture"):
                self.video_capture.release()
        except Exception:
            pass
        super().closeEvent(event)

    def resizeEvent(self, e):
        super().resizeEvent(e)



