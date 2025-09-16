import sys
import time
import threading
import queue
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QSizePolicy, QCheckBox, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


class CaptureThread(threading.Thread):
    def __init__(self, src, cap2det_q, stop_event):
        super().__init__(daemon=True)
        self.src = src
        self.cap2det_q = cap2det_q
        self.stop_event = stop_event
        self.cap = cv2.VideoCapture(src)

    def run(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            try:
                self.cap2det_q.put_nowait(frame)
            except queue.Full:
                try:
                    _ = self.cap2det_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.cap2det_q.put_nowait(frame)
                except queue.Full:
                    pass
        self.cap.release()


class DetectThread(threading.Thread):
    def __init__(self, model_path, img_size, yolo_rate,
                 classes, colors, roi_check,
                 enable_flags, cap2det_q, det2ui_q, stop_event):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.img_size = img_size
        self.yolo_rate = yolo_rate
        self.classes = classes
        self.colors = colors
        self.roi_check = roi_check or [-1, 9999, -1, 9999]
        self.enable_flags = enable_flags  # dict: fell, helmet, jacket, fire, smoke
        self.cap2det_q = cap2det_q
        self.det2ui_q = det2ui_q
        self.stop_event = stop_event

    def run(self):
        model = YOLO(self.model_path) if self.model_path else YOLO("yolov8n.pt")
        x1, x2, y1, y2 = self.roi_check
        while not self.stop_event.is_set():
            try:
                frame = self.cap2det_q.get(timeout=0.05)
            except queue.Empty:
                continue

            results = model.predict(frame, imgsz=self.img_size, verbose=False)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotator = Annotator(rgb, line_width=1, font_size=16)

            is_warning = False
            for r in results:
                boxes = getattr(r, 'boxes', [])
                for box in boxes:
                    try:
                        conf_ok = float(box.conf) > float(self.yolo_rate)
                    except Exception:
                        conf_ok = True
                    if not conf_ok:
                        continue
                    b = box.xyxy[0]
                    try:
                        bx1, by1, bx2, by2 = [float(v) for v in b]
                    except Exception:
                        # fallback for unexpected types
                        bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                    if bx1 < x1 or by1 < y1 or bx2 > x2 or by2 > y2:
                        continue
                    c = int(box.cls)

                    helmet_ok = (c == 6 and self.enable_flags.get('helmet', False))
                    fell_ok = (c == 5 and self.enable_flags.get('fell', False))
                    jacket_ok = (c == 7 and self.enable_flags.get('jacket', False))
                    fire_ok = (c == 8 and self.enable_flags.get('fire', False))
                    smoke_ok = (c == 9 and self.enable_flags.get('smoke', False))

                    if helmet_ok or fell_ok or jacket_ok or fire_ok or smoke_ok:
                        is_warning = True
                        annotator.box_label([bx1, by1, bx2, by2], self.classes[c] if c < len(self.classes) else str(c),
                                            color=tuple(self.colors[int(c)]) if int(c) < len(self.colors) else (255, 0, 0))
                        continue
                    annotator.box_label([bx1, by1, bx2, by2], self.classes[c] if c < len(self.classes) else str(c), color=(0, 255, 0))

            img = annotator.result()
            h, w, _ = img.shape
            if is_warning:
                annotator.box_label([5, 5, w - 5, h - 5], "", color=(255, 0, 0))
                img = annotator.result()

            payload = (img, is_warning)
            try:
                self.det2ui_q.put_nowait(payload)
            except queue.Full:
                try:
                    _ = self.det2ui_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.det2ui_q.put_nowait(payload)
                except queue.Full:
                    pass


class CameraWidget(QWidget):
    def __init__(self, camera_name="Camera", camera_src=0, img_size=640, yolo_model_path="",
                 yolo_rate=0.5, is_fell_check=True, is_fire_check=False, is_smoke_check=False,
                 is_heltmet_check=False, is_jacket_check=False, timer_delay=20,
                 classes=("", "Helmet", "Jacket", "_3", "_4", "Fall", "No Helmet", "No Jacket", "Fire", "Smoke"),
                 colors=((0,0,255),(0,255,0),(255,0,255),(0,0,0),(0,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0)),
                 roi_check=None,
                 parent=None):
        super().__init__(parent)

        self.camera_name = camera_name
        self.img_size = img_size
        self.yolo_model_path = yolo_model_path
        self.yolo_rate = yolo_rate
        self.classes = list(classes)
        self.colors = list(colors)
        self.roi_check = roi_check or [-1, 9999, -1, 9999]
        self.is_warning = False

        # Queues between threads
        self.cap2det_q = queue.Queue(maxsize=2)   # small, drop when full
        self.det2ui_q = queue.Queue(maxsize=1)    # UI only needs the latest

        # Stop event for clean shutdown
        self.stop_event = threading.Event()

        # UI setup
        self.layout = QGridLayout(self)
        self.image_label = QLabel(self)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.image_label, 0, 0, 3, 3)

        self.camera_name_label = QLabel(self)
        self.camera_name_label.setStyleSheet("font-size: 20px; font-weight: bold; color: white; padding: 4px;")
        self.camera_name_label.setText(camera_name)
        self.layout.addWidget(self.camera_name_label, 0, 0, 1, 1)

        self.label_warning = QLabel(self)
        self.label_warning.setText("WARNING")
        self.label_warning.setStyleSheet("font-size: 48px; font-weight: bold; color: red;")
        self.label_warning.setVisible(False)
        self.layout.addWidget(self.label_warning, 1, 1)

        # Control checkboxes (lightweight)
        self.checkbox_layout = QVBoxLayout(self)
        self.checkbox_helmet = QCheckBox("Helmet")
        self.checkbox_helmet.setChecked(is_heltmet_check)
        self.checkbox_fell = QCheckBox("Fell")
        self.checkbox_fell.setChecked(is_fell_check)
        self.checkbox_jacket = QCheckBox("Jacket")
        self.checkbox_jacket.setChecked(is_jacket_check)
        self.checkbox_fire = QCheckBox("Fire")
        self.checkbox_fire.setChecked(is_fire_check)
        self.checkbox_smoke = QCheckBox("Smoke")
        self.checkbox_smoke.setChecked(is_smoke_check)
        for cb in (self.checkbox_fell, self.checkbox_helmet, self.checkbox_jacket, self.checkbox_fire, self.checkbox_smoke):
            cb.setStyleSheet("font-size: 14px; color: white;")
            self.checkbox_layout.addWidget(cb)
        self.layout.addLayout(self.checkbox_layout, 0, 2)

        self.setLayout(self.layout)

        # Threads
        src = self._normalize_src(camera_src)
        self.cap_thread = CaptureThread(src, self.cap2det_q, self.stop_event)
        enable_flags = {
            'fell': is_fell_check,
            'helmet': is_heltmet_check,
            'jacket': is_jacket_check,
            'fire': is_fire_check,
            'smoke': is_smoke_check,
        }
        self.det_thread = DetectThread(self.yolo_model_path, self.img_size, self.yolo_rate,
                                       self.classes, self.colors, self.roi_check,
                                       enable_flags, self.cap2det_q, self.det2ui_q, self.stop_event)

        self.cap_thread.start()
        self.det_thread.start()

        # UI timer to pull processed frames
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._pull_and_show)
        self.ui_timer.start(max(10, int(timer_delay)))

        # Blink timer for warning label (lightweight)
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self._blink_warning)
        self.blink_timer.start(500)

    def _normalize_src(self, camera_src):
        if isinstance(camera_src, int):
            return camera_src
        try:
            return int(camera_src)
        except Exception:
            return camera_src

    def _pull_and_show(self):
        last = None
        while not self.det2ui_q.empty():
            try:
                last = self.det2ui_q.get_nowait()
            except queue.Empty:
                break
        if last is None:
            return
        img, is_warning = last
        self.is_warning = is_warning
        h, w, ch = img.shape
        q_img = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _blink_warning(self):
        if self.is_warning:
            self.label_warning.setVisible(not self.label_warning.isVisible())
        else:
            self.label_warning.setVisible(False)

    def closeEvent(self, event):
        self.stop_event.set()
        try:
            self.cap_thread.join(timeout=1.0)
            self.det_thread.join(timeout=1.0)
        except Exception:
            pass
        super().closeEvent(event)

