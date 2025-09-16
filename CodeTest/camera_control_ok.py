
import queue
import threading
import time

from PIL import Image
import io

import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter, QPen
from PyQt5.QtCore import QTimer, QSize
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from datetime import datetime
import base64
from unidecode import unidecode

from WidgetCustom import *
from queue import Queue
import gc
from display_image import *

import os
from pathlib import Path


class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
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
        return self.q.get()

    def release(self):
        self.cap.release()



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
        self.frame_camera = None
        self.video_writer = None

        i = int(camera_src) if isinstance(camera_src, int) else camera_src
        print(i)
        # self.video_capture = VideoCapture(i)
        self.capture = VideoCapture(i)


        # self.queueImage = Queue()

        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.record_frame)
        # self.timer.start(1)  # Cập nhật mỗi 10ms

        self.timer_watch = QTimer(self)
        self.timer_watch.timeout.connect(self.update_date_time)
        self.timer_watch.start(1000)


        # self.q = queue.Queue()
        t = threading.Thread(target=self.autoTest)
        t.daemon = True
        t.start()

    def initUI(self, camera_name="None"):

        # Grid tổng
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)
        # Label hiển thị ảnh
        self.image_label = QLabel(self)

        # self.image_label.setStyleSheet("border :5px solid green;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.layout.addWidget(self.image_label, 0, 0, 3, 3)  # Đặt label vào ô (0, 1)

        #self.image_label.setPixmap(QPixmap.fromImage(self.frame_1).scaled(self.image_label.size()))
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
        self.layout.addWidget(self.label_warning, 1, 1,Qt.AlignCenter)

        # Set chiều rộng, cao cho các grid
        self.layout.setRowMinimumHeight(2, 50)
        self.layout.setRowMinimumHeight(0, 50)
   

        self.layout.addLayout(self.checkbox_layout, 0, 2)  # Đặt checkbox vào ô (0, 0)
        # Đặt layout chính cho widget
        self.setLayout(self.layout)

        self.image_folder_path = unidecode(self.camera_name).replace(" ", "")
        # creat save image folder
        # if not os.path.exists(self.image_folder_path):
        #     os.makedirs(self.image_folder_path)
        self.frame_count = 0
        self.frame_camera = None
        self.is_running = False

    def encode_frame(self, file_path_encode, frame):

        # cv2.imwrite(file_path,frame)
        frame_encode = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert frame to object PIL
        image = Image.fromarray(frame_encode)

        # Convert image to byte format
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_byte = buffered.getvalue()

        # encode byte to base64
        img_base64 = base64.b64encode(img_byte).decode('utf-8')

        # Save string base64 to file txt
        with open(file_path_encode, 'w') as f:
            f.write(img_base64)
        # print("Save image warning DONE")

    def autoTest(self):

        while True:
            try:
                frame = self.capture.read()
                x1, x2, y1, y2 = self.roi_check
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                results = self.model.predict(frame, imgsz=self.img_size)

                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotator = Annotator(cv2image, line_width=1, font_size=16)

                result = True
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        if box.conf > self.yolo_rate:
                            b = box.xyxy[0]
                            c = int(box.cls)
                            # Fell check
                            if ((c == 5 and self.is_fell_check) or (c == 6 and self.is_helmet_check) or (
                                    c == 7 and self.is_jacket_check)
                                    or (c == 8 and self.is_fire_check) or (c == 9 and self.is_smoke_check)):
                                # Save image warning
                                file_path_encode = "./image_encode/" + self.image_folder_path + "_" + datetime.now().strftime(
                                    "%Y%m%d%H%M%S") + "_warning.txt"

                                self.encode_frame(file_path_encode, frame)

                   
                                result = False

                                annotator.box_label(b, self.classes[c], color=self.colors[int(c)])
                                time.sleep(0.005)
                                continue
                            time.sleep(0.005)
                            annotator.box_label(b, self.classes[c], color=(0, 255, 0))

                self.is_warning = not result
                img = annotator.result()
                h, w, ch = img.shape
                if not result:

                    annotator_box = Annotator(cv2image, line_width=5, font_size=16)
                    annotator_box.box_label([5, 5, w - 5, h - 5], "", color=(255, 0, 0))
                bytes_per_line = ch * w
                q_image = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.image_label.setPixmap(QPixmap.fromImage(q_image).scaled(self.image_label.size()))
                time.sleep(0.005)
            except Exception as e:
                print(f"Exception {e}")

    def onHelmetStateChange(self, state):
        if state == 2:  # Qt.Checked = 2
            self.is_helmet_check = True
        else:
            self.is_helmet_check = False

    def onFellStateChange(self, state):
        if state == 2:  # Qt.Checked = 2
            self.is_fell_check = True
        else:
            self.is_fell_check = False

    def onJacketStateChange(self, state):
        if state == 2:  # Qt.Checked = 2
            self.is_jacket_check = True
        else:
            self.is_jacket_check = False

    def onFireStateChange(self, state):
        if state == 2:  # Qt.Checked = 2
            self.is_fire_check = True
        else:
            self.is_fire_check = False

    def onSmokeStateChange(self, state):
        if state == 2:  # Qt.Checked = 2
            self.is_smoke_check = True
        else:
            self.is_smoke_check = False

    def update_date_time(self):
        self.label_date_time.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        if self.is_warning:
            self.label_warning.setVisible(not self.label_warning.isVisible())
        else:
            self.label_warning.setVisible(False)



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

        i = int(camera_src) if isinstance(camera_src, int) else camera_src
        print(i)
        self.video_capture = VideoCapture(i)
        #self.capture = cv2.VideoCapture(i)


        #self.queueImage = Queue()

        # tupdateframe = threading.Thread(target=self.update_frame,args=(i,))
        # tupdateframe.start()

        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.update_frame)
        # self.timer.start(self.timer_delay)  # Cập nhật mỗi 10ms

        self.timer_watch = QTimer(self)
        self.timer_watch.timeout.connect(self.update_date_time)
        self.timer_watch.start(1000)

        # self.q = queue.Queue()
        t = threading.Thread(target=self.autoTest)
        # t.daemon = True
        t.start()

    def initUI(self, camera_name="None"):

        # Grid tổng
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)
        # Label hiển thị ảnh
        self.image_label = QLabel(self)

        # self.image_label.setStyleSheet("border :5px solid green;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.layout.addWidget(self.image_label, 0, 0, 3, 3)  # Đặt label vào ô (0, 1)
        painter = QPainter(self.image_label)
        pen = QPen(Qt.red, 12)  # Thiết lập bút màu đỏ với độ dày là 2
        painter.setPen(pen)

        # Vẽ hình chữ nhật
        painter.drawRect(self.image_label.rect())
        self.image_label.update()

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
        self.checkbox_helmet.setStyleSheet("font-size: 16px; font-weight: bold; padding : 3px;  color : white;")
        self.checkbox_helmet.setChecked(self.is_helmet_check)
        self.checkbox_helmet.stateChanged.connect(self.onHelmetStateChange)

        # Checkbox Fell check
        self.checkbox_fell = QCheckBox("Fell")
        self.checkbox_fell.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 3px; margin-top: 0px;color: white;")
        self.checkbox_fell.setChecked(self.is_fell_check)
        self.checkbox_fell.stateChanged.connect(self.onFellStateChange)

        # Checkbox Jacket check
        self.checkbox_jacket = QCheckBox("Jacket")
        self.checkbox_jacket.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 3px; margin-top: 0px;color: white;")
        self.checkbox_jacket.setChecked(self.is_jacket_check)
        self.checkbox_jacket.stateChanged.connect(self.onJacketStateChange)

        # Checkbox Fire check
        self.checkbox_fire = QCheckBox("Fire")
        self.checkbox_fire.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 3px; margin-top: 0px;color: white;")
        self.checkbox_fire.setChecked(self.is_fire_check)
        self.checkbox_fire.stateChanged.connect(self.onFireStateChange)

        # Checkbox Smoke check
        self.checkbox_smoke = QCheckBox("Smoke")
        self.checkbox_smoke.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 3px; margin-top: 0px;color: white;")
        self.checkbox_smoke.setChecked(self.is_smoke_check)
        self.checkbox_smoke.stateChanged.connect(self.onSmokeStateChange)

        self.checkbox_layout.addWidget(self.checkbox_fell)
        self.checkbox_layout.addWidget(self.checkbox_helmet)
        self.checkbox_layout.addWidget(self.checkbox_jacket)
        self.checkbox_layout.addWidget(self.checkbox_fire)
        self.checkbox_layout.addWidget(self.checkbox_smoke)
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
        # self.label_warning = QVBoxLayout(frame)
        # self.layout.setContentsMargins(0,0,40,0)
        self.label_warning.setVisible(False)
        # self.label_warning.setContentsMargins(0, 0, 0, 0)
        self.label_warning.setStyleSheet(
            "font-size: 85px; font-weight: bold; color : red; padding-right: 100px;padding-bottom: 100px")
        self.layout.addWidget(self.label_warning, 1, 1)

        # Set chiều rộng, cao cho các grid
        self.layout.setRowMinimumHeight(2, 80)
        # self.layout.setRowMinimumHeight(0, 50)
        self.layout.setColumnMinimumWidth(0, 50)
        self.layout.setColumnMinimumWidth(2, 50)

        self.layout.addLayout(self.checkbox_layout, 0, 2)  # Đặt checkbox vào ô (0, 0)
        # Đặt layout chính cho widget
        self.setLayout(self.layout)

        self.image_folder_path = unidecode(self.camera_name).replace(" ", "")
        # creat save image folder
        # if not os.path.exists(self.image_folder_path):
        #     os.makedirs(self.image_folder_path)
        self.frame_count = 0
        self.frame_camera = None
        self.is_running = False

    def encode_frame(self, file_path_encode, frame):

        # cv2.imwrite(file_path,frame)
        frame_encode = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert frame to object PIL
        image = Image.fromarray(frame_encode)

        # Convert image to byte format
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_byte = buffered.getvalue()

        # encode byte to base64
        img_base64 = base64.b64encode(img_byte).decode('utf-8')

        # Save string base64 to file txt
        with open(file_path_encode, 'w') as f:
            f.write(img_base64)
        # print("Save image warning DONE")

    def autoTest(self):

        while True:
            # file_path = r"./data_fail/" + self.image_folder_path + "/" + datetime.now().strftime("%Y%m%d%H%M%S") + "_warning.jpg"
            try:
                #frame = self.queueImage.get()
                frame = self.video_capture.read()

                x1, x2, y1, y2 = self.roi_check
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                results = self.model.predict(frame, imgsz=self.img_size)

                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotator = Annotator(cv2image, line_width=1, font_size=16)

                result = True
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        if box.conf > self.yolo_rate:
                            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                            if b[0] < x1 or b[1] < y1 or b[2] > x2 or b[3] > y2:
                                continue
                            c = int(box.cls)
                            # Fell check
                            if ((c == 5 and self.is_fell_check) or (c == 6 and self.is_helmet_check) or (
                                    c == 7 and self.is_jacket_check)
                                    or (c == 8 and self.is_fire_check) or (c == 9 and self.is_smoke_check)):
                                # Save image warning
                                file_path_encode = "./image_encode/" + self.image_folder_path + "_" + datetime.now().strftime(
                                    "%Y%m%d%H%M%S") + "_warning.txt"

                                #self.encode_frame(file_path_encode, frame)

                                # self.decode_frame(file_path_encode)

                                # cv2.imwrite(file_path,frame)
                                result = False

                                annotator.box_label(b, self.classes[c], color=self.colors[int(c)])
                                time.sleep(0.005)
                                continue

                            time.sleep(0.005)
                            annotator.box_label(b, self.classes[c], color=(0, 255, 0))

                self.is_warning = not result
                img = annotator.result()
                h, w, ch = img.shape

                # hiển thị ô màu đỏ cảnh báo
                if not result:
                    annotator.box_label([5, 5, w - 5, h - 5], "", color=(255, 0, 0))
                bytes_per_line = ch * w
                q_image = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.image_label.setPixmap(QPixmap.fromImage(q_image).scaled(self.image_label.size()))
                gc.collect()
                # time.sleep()
            except Exception as e:
                print(f"Exception {e}")

    def onHelmetStateChange(self, state):
        if state == 2:  # Qt.Checked = 2
            self.is_helmet_check = True
        else:
            self.is_helmet_check = False

    def onFellStateChange(self, state):
        if state == 2:  # Qt.Checked = 2
            self.is_fell_check = True
        else:
            self.is_fell_check = False

    def onJacketStateChange(self, state):
        if state == 2:  # Qt.Checked = 2
            self.is_jacket_check = True
        else:
            self.is_jacket_check = False

    def onFireStateChange(self, state):
        if state == 2:  # Qt.Checked = 2
            self.is_fire_check = True
        else:
            self.is_fire_check = False

    def onSmokeStateChange(self, state):
        if state == 2:  # Qt.Checked = 2
            self.is_smoke_check = True
        else:
            self.is_smoke_check = False

    def update_frame(self):
        while True:
            ret, frame = self.capture.read()
            # print(f"Capture Result  {ret}")
            if ret:

                self.frame_camera = frame
                if self.queueImage.full():
                    self.queueImage.get_nowait()
                self.queueImage.put(frame)
                continue

    def update_date_time(self):
        self.label_date_time.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        if self.is_warning:
            self.label_warning.setVisible(not self.label_warning.isVisible())
        else:
            self.label_warning.setVisible(False)
