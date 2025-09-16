
import queue
import threading
import time

from PIL import Image
import io

import cv2
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

import os,re
from pathlib import Path


INVALID_CHARS = r'[^A-Za-z0-9_.-]'
def safe_name(s: str) -> str:
    """Biến chuỗi thành tên file an toàn cho Windows."""
    s = os.path.basename(str(s))      # nếu là path, chỉ lấy tên cuối
    s = s.replace(':', '_')           # bỏ dấu :
    s = re.sub(INVALID_CHARS, '_', s) # thay kí tự lạ
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
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("FAIL READ FRAME")
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
        self.cap.release()



class CameraThread(QThread):
    frame_captured = pyqtSignal(QImage)
    warning_changed = pyqtSignal(bool)

    def __init__(self, camera_id, model, yolo_rate, img_size, roi_check, classes, colors,
                 enable_flags,camera_name):
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
        self.save_dir = Path("./image")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._last_warn_ts = 0.0
        self._last_warning_state = False

    def run(self):
        x1, x2, y1, y2 = self.roi_check
        while self.running:
            frame = self.video_capture.read()
            if frame is None:
                time.sleep(0.01)
                continue
            results = self.model.predict(frame, imgsz=self.img_size)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotator = Annotator(cv2image, line_width=1, font_size=16)
            is_warning = False
            for r in results:
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
                    c = int(box.cls)
                    helmet_ok = (c == 55 and self.enable_flags.get('helmet', False))
                    fell_ok = (c == 57 and self.enable_flags.get('fell', False))
                    jacket_ok = (c == 74 and self.enable_flags.get('jacket', False))
                    fire_ok = (c == 84 and self.enable_flags.get('fire', False))
                    smoke_ok = (c == 93 and self.enable_flags.get('smoke', False))
                    label = self.classes[c] if c < len(self.classes) else str(c)
                    if helmet_ok or fell_ok or jacket_ok or fire_ok or smoke_ok or c == 2:
                        is_warning = True
                        try:
                            os.makedirs("./image_encode", exist_ok=True)
                            file_path_encode = "./image_encode/" + self.image_folder_path + "_" + datetime.now().strftime(
                                "%Y%m%d%H%M%S") + "_warning.txt"
                            self.encode_frame(file_path_encode, frame)
                        except Exception:
                            pass
                        color = tuple(self.colors[int(c)]) if int(c) < len(self.colors) else (255, 0, 0)
                        annotator.box_label([bx1, by1, bx2, by2], label, color=color)
                    else:
                        annotator.box_label([bx1, by1, bx2, by2], label, color=(0, 255, 0))
            img = annotator.result()
            h, w, _ = img.shape
            if is_warning:
                annotator.box_label([5, 5, w - 5, h - 5], "", color=(255, 0, 0))
                img = annotator.result()
                now_ts = time.time()
                if now_ts - self._last_warn_ts > 10.0:
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

        # Watch timer for clock + blink
        self.timer_watch = QTimer(self)
        self.timer_watch.timeout.connect(self.update_date_time)
        self.timer_watch.start(1000)

    def initUI(self, camera_name="None"):

        # Grid tổng
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)
        # Label hiển thị ảnh
        self.image_label = QLabel(self)

        # self.image_label.setStyleSheet("border :5px solid green;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.layout.addWidget(self.image_label, 0, 0, 3, 3)  # Đặt label vào ô (0, 1)

        # self.image_label.setPixmap(QPixmap.fromImage(self.frame_1).scaled(self.image_label.size()))
        # pix = self.image_label.pixmap().copy()
        painter = QPainter(self.image_label)
        pen = QPen(QColor(0, 255, 255), 12)  # Thiết lập bút màu đỏ với độ dày là 2
        painter.setPen(pen)

        self.image_label.update()
        painter.end()


        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

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
            "font-size: 65px; font-weight: bold; color : red;")
        self.layout.addWidget(self.label_warning, 1, 1 ,Qt.AlignCenter)

        # Set chiều rộng, cao cho các grid
        self.layout.setRowMinimumHeight(2, 50)
        self.layout.setRowMinimumHeight(0, 50)


        self.layout.addLayout(self.checkbox_layout, 0, 2)  # Đặt checkbox vào ô (0, 0)
        # Đặt layout chính cho widget
        self.setLayout(self.layout)

        self.image_folder_path = unidecode(self.camera_name).replace(" ", "")

        self.frame_count = 0


    def _on_frame(self, q_image: QImage):
        self.image_label.setPixmap(QPixmap.fromImage(q_image).scaled(self.image_label.size()))

    def _on_warning_changed(self, flag: bool):
        self.is_warning = flag

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



