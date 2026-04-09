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
from ffmpeg_capture import FFmpegCapture


class VideoCapture:

    def __init__(self, name, logger: Logger = None):
        self.logger = logger
        self.source = name
        self.is_file = isinstance(name, str) and os.path.isfile(name)
        self.cap = None
        self.running = True
        self._last_frame_ok_ts = 0.0
        self._stall_timeout_sec = 5.0
        self._reconnect_delay_sec = 2.0
        self._reconnect_delay_max_sec = 30.0
        self._ffmpeg: FFmpegCapture | None = None

        # Nếu là RTSP và ffmpeg khả dụng: ưu tiên FFmpegCapture (ổn định 24/7)
        if isinstance(self.source, str) and self.source.lower().startswith("rtsp") and FFmpegCapture.is_available():
            try:
                # target_width có thể set theo imgsz nếu bạn muốn; tạm dùng None để giữ nguyên
                self._ffmpeg = FFmpegCapture(self.source, target_width=None, logger=self.logger)
            except Exception as e:
                self._ffmpeg = None
                if self.logger:
                    self.logger.warning(f"FFmpegCapture init failed, fallback OpenCV: {e}")

        # Fallback OpenCV capture
        if self._ffmpeg is None:
            # mở kết nối lần đầu
            self._open_capture()
            self.q = queue.Queue(maxsize=1)
            self._t = threading.Thread(target=self._reader, name=f"VideoReader-{self.source}", daemon=True)
            self._t.start()
        else:
            self.q = None
            self._t = None

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        """
        Vòng lặp đọc frame với cơ chế reconnect cho RTSP:
        - Nếu mất kết nối (ret=False), thử mở lại sau một khoảng delay.
        - Tránh break hẳn thread để RTSP tạm mất tín hiệu không làm crash app.
        """
        reconnect_delay = float(self._reconnect_delay_sec)
        while self.running:
            # đảm bảo cap đang mở; nếu không, thử mở lại
            if self.cap is None or not self.cap.isOpened():
                self._open_capture()
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2.0, float(self._reconnect_delay_max_sec))
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
                reconnect_delay = min(reconnect_delay * 2.0, float(self._reconnect_delay_max_sec))
                continue
            # reset backoff khi đã đọc được frame
            reconnect_delay = float(self._reconnect_delay_sec)
            self._last_frame_ok_ts = time.monotonic()

            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)
        
            # Không check stall ở đây (vì vừa có frame OK); stall sẽ được phát hiện bởi nhánh read() fail/timeout.

    def read(self):
        if self._ffmpeg is not None:
            return self._ffmpeg.read(timeout=0.5)
        try:
            return self.q.get(timeout=0.5)
        except queue.Empty:
            return None

    def release(self):
        try:
            self.running = False
            if self._ffmpeg is not None:
                self._ffmpeg.release()
                self._ffmpeg = None
            if self.cap is not None:
                self.cap.release()
        finally:
            if hasattr(self, '_t') and self._t is not None and self._t.is_alive():
                try:
                    self._t.join(timeout=0.2)
                except Exception:
                    pass

    def _open_capture(self):
        """Mở hoặc mở lại VideoCapture với cấu hình phù hợp (RTSP/USB/file)."""
        try:
            # Ưu tiên FFmpeg backend cho RTSP để ổn định hơn (nếu build OpenCV có FFmpeg)
            if isinstance(self.source, str) and self.source.lower().startswith("rtsp"):
                try:
                    self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                except Exception:
                    self.cap = cv2.VideoCapture(self.source)
            else:
                self.cap = cv2.VideoCapture(self.source)

            # set timeout nếu OpenCV build hỗ trợ (không phải build nào cũng có)
            try:
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            except Exception:
                pass
            try:
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            except Exception:
                pass

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
        # trạng thái tín hiệu camera (dựa trên việc có frame mới gần đây)
        self._has_signal = False
        self._last_frame_ts = 0.0
        self._signal_timeout_sec = 30.0
        self.initUI(camera_name)
        self.logger = logger

        # Engine YOLO dùng chung
        self.engine = get_yolo_engine()

        # Video capture riêng cho từng camera (thread nhẹ đọc frame)
        self.video_capture = VideoCapture(camera_src, logger=self.logger)

        # Cờ và tham số điều phối FPS gửi vào YOLO
        self._infer_in_flight = False
        # Mặc định ưu tiên ổn định RTSP 24/7 cho 4–6 camera
        self._target_fps = 6.0  # tối đa 6 inference/giây cho mỗi camera
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
        # Quan trọng: tránh QLabel thay đổi sizeHint theo kích thước pixmap
        # (nguyên nhân phổ biến làm widget “phóng to dần” khi setPixmap liên tục).
        self.image_label.setScaledContents(False)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(1, 1)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.layout.addWidget(self.image_label, 0, 0, 3, 3)  # Đặt label vào ô (0, 1)

        # Label hiển thị tên camera
        self.camera_name_label = QLabel(self)
        self.camera_name_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; padding-left: 5px;"
            "padding-right: 0px; margin-top: 6px; padding-top: 0px; color: #f0f0f0;")
        self.camera_name_label.setText(camera_name)
        # Name luôn ở góc trên trái, span 2 cột để chừa không gian checkbox bên phải
        self.layout.addWidget(self.camera_name_label, 0, 0, 1, 2, Qt.AlignTop | Qt.AlignLeft)

        # Trạng thái LIVE / NO SIGNAL (chấm + text)
        self.status_widget = QWidget(self)
        status_layout = QHBoxLayout(self.status_widget)
        status_layout.setContentsMargins(6, 0, 0, 0)
        status_layout.setSpacing(6)

        self.status_dot = QLabel(self.status_widget)
        self.status_dot.setFixedSize(10, 10)
        self.status_dot.setStyleSheet("background-color: #f0c000; border-radius: 5px;")  # default yellow

        self.status_text = QLabel("NO SIGNAL", self.status_widget)
        self.status_text.setStyleSheet("color: #f0c000; font-weight: bold; font-size: 12px;")

        status_layout.addWidget(self.status_dot)
        status_layout.addWidget(self.status_text)
        status_layout.addStretch(1)

        # Status nằm ngay dưới tên camera
        self.layout.addWidget(self.status_widget, 1, 0, 1, 2, Qt.AlignTop | Qt.AlignLeft)

        # Layout hiển tḥ check box
        self.checkbox_layout = QVBoxLayout(self)
        # đẩy xuống một chút để không dính sát mép trên
        self.checkbox_layout.setContentsMargins(0, 6, 0, 0)
        self.checkbox_layout.setSpacing(4)

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

        # Overlay NO SIGNAL
        self.label_no_signal = QLabel()
        self.label_no_signal.setText("NO SIGNAL")
        self.label_no_signal.setAlignment(Qt.AlignHCenter)
        self.label_no_signal.setAlignment(Qt.AlignVCenter)
        self.label_no_signal.setVisible(True)
        self.label_no_signal.setContentsMargins(0, 0, 0, 0)
        self.label_no_signal.setStyleSheet(
            "font-size: 32px; font-weight: bold; color: #f0c000;"
            "background-color: rgba(0, 0, 0, 180); padding: 8px;"
        )
        self.layout.addWidget(self.label_no_signal, 1, 1, 1, 1, Qt.AlignCenter)

        # Set chiều rộng, cao cho các grid
        self.layout.setRowMinimumHeight(2, 50)
        self.layout.setRowMinimumHeight(0, 50)

        # Checkbox luôn giữ ở góc trên bên phải
        self.layout.addLayout(self.checkbox_layout, 0, 2, 2, 1, Qt.AlignTop | Qt.AlignRight)
        # Đặt layout chính cho widget

        self.image_folder_path = unidecode(self.camera_name).replace(" ", "")

        self.frame_count = 0

    def _on_frame_timer(self):
        """Đọc frame mới và (nếu đủ điều kiện) gửi request inference vào YoloEngine."""
        frame = self.video_capture.read()
        if frame is None:
            # nếu quá timeout thì báo mất tín hiệu
            now_ts = time.monotonic()
            if self._last_frame_ts <= 0 or (now_ts - self._last_frame_ts) > float(self._signal_timeout_sec):
                self._set_signal_state(False)
            return

        # có frame -> live
        self._last_frame_ts = time.monotonic()
        if not self._has_signal:
            self._set_signal_state(True)

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

    def _set_signal_state(self, has_signal: bool):
        """Cập nhật UI trạng thái tín hiệu: LIVE (green) / NO SIGNAL (yellow)."""
        self._has_signal = bool(has_signal)
        try:
            if self._has_signal:
                if hasattr(self, "status_dot") and self.status_dot:
                    self.status_dot.setStyleSheet("background-color: #2ecc71; border-radius: 5px;")
                if hasattr(self, "status_text") and self.status_text:
                    self.status_text.setText("LIVE")
                    self.status_text.setStyleSheet("color: #2ecc71; font-weight: bold; font-size: 12px;")
                if hasattr(self, "label_no_signal") and self.label_no_signal:
                    self.label_no_signal.setVisible(False)
            else:
                if hasattr(self, "status_dot") and self.status_dot:
                    self.status_dot.setStyleSheet("background-color: #f0c000; border-radius: 5px;")
                if hasattr(self, "status_text") and self.status_text:
                    self.status_text.setText("NO SIGNAL")
                    self.status_text.setStyleSheet("color: #f0c000; font-weight: bold; font-size: 12px;")
                if hasattr(self, "label_no_signal") and self.label_no_signal:
                    self.label_no_signal.setVisible(True)
                # mất tín hiệu thì không hiển thị WARNING
                if hasattr(self, "label_warning") and self.label_warning:
                    self.label_warning.setVisible(False)
        except Exception:
            pass

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



