import sys
import threading
import time
from PIL import Image
import io

import cv2
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter, QPen
from PyQt5.QtCore import QTimer, QSize

from datetime import datetime
import base64
import numpy as np
from unidecode import unidecode

from WidgetCustom import *
from queue import Queue

import os
from pathlib import Path


class LogWidget(QWidget):
    def __init__(self, camera_name="None", img_size=640,
                 colors=[(0, 0, 255), (0, 255, 0), (255, 0, 0)], timer_delay=10, parent=None, directory=None,
                 max_items=10, list_image=[]):
        super().__init__(parent)
        self.frame_count = None
        self.camera_name = camera_name
        self.img_size = img_size
        self.colors = colors

        self.timer_delay = timer_delay
        self.is_warning = False
        self.initUI(camera_name)

        self.directory = directory
        self.max_items = max_items
        self.list_image = list_image

        # self.timer_watch = QTimer(self)
        # self.timer_watch.timeout.connect(self.update_date_time)
        # self.timer_watch.start(1000)  # Cập nhật mỗi 10ms

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.decode_frame)
        self.timer.start(10000)  # Cập nhật mỗi 1s

        self.timer_last = QTimer(self)
        self.timer_last.timeout.connect(self.delete_file_encode)
        self.timer_last.start(10000)  # Cập nhật mỗi 1s

    def initUI(self, camera_name="None"):

        # Grid tổng
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(2, 2, 0, 2)
        # Label hiển thị ảnh
        self.image_label = QLabel(self)
        self.image_label.setStyleSheet("margin: 0px; padding : 0px;")

        # self.image_label.setStyleSheet("border :5px solid green;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.layout.addWidget(self.image_label, 0, 0, 3, 3)  # Đặt label vào ô (0, 1)


        # Vẽ hình chữ nhật
        # painter.drawRect(self.image_label.rect())
        self.image_label.update()
        # self.layout.addWidget(self.image_label,0,0,2,2)

        # Label hiển thị tên
        self.camera_name_label = QLabel(self)
        self.camera_name_label.setStyleSheet("font-size: 14px; font-weight: bold; padding : 5px;  color : white;")
        self.camera_name_label.setText(camera_name)
        # self.layout.setRowMinimumHeight(0, 80)
        self.layout.addWidget(self.camera_name_label, 0, 0)


        # Set chiều rộng, cao cho các grid

        self.layout.setColumnMinimumWidth(2, 50)

        self.setLayout(self.layout)

        self.image_folder_path = unidecode(self.camera_name).replace(" ", "")
        # creat save image folder
        # if not os.path.exists(self.image_folder_path):
        #     os.makedirs(self.image_folder_path)
        self.frame_count = 0
        self.frame_camera = None
        self.is_running = False

    def delete_file_encode(self):
        image_path_decode = "./image_decode/"
        files = os.listdir(self.directory)
        files = [f for f in files if os.path.isfile(os.path.join(self.directory, f))]
        if len(files) > 50:
            files.sort(key=lambda x: os.path.getmtime(os.path.join(self.directory, x)), reverse=True)
            for file_name in files[19:]:
                image_path_decode = "./image_encode/" + file_name
                print(image_path_decode)
                os.remove(image_path_decode)

    def decode_frame(self):

        file_name = self.last_detector_warning()
        if file_name == None:
            return
        with open(file_name, 'r') as text_file:
            encoded_string = text_file.read()
        # print(encoded_string)
        image_data = base64.b64decode(encoded_string)

        # Chuyển đổi bytes thành QImage
        image = QImage()
        image.loadFromData(image_data)

        pixmap = QPixmap.fromImage(image)
        resized_pixmap = pixmap.scaled(400, 300)

        resized_pixmap.size()
        # self.image_label.resize(640, 320)
        self.image_label.setPixmap(resized_pixmap)



    def last_detector_warning(self):
        files = os.listdir(self.directory)
        files = [f for f in files if os.path.isfile(os.path.join(self.directory, f))]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(self.directory, x)), reverse=True)
        if files == None or len(files) == 0:
            return
        file_name = os.path.join(self.directory, files[0])
        return file_name

    def capture_warning(self):
        return 0

    def closeEvent(self, event):
        self.capture.release()
        super().closeEvent(event)
