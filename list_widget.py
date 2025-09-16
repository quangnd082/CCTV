import threading
import time
from PIL import Image
import io
from io import BytesIO
import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter, QPen
from PyQt5.QtCore import QTimer, QSize

from datetime import datetime
import base64
import numpy as np
from unidecode import unidecode

from WidgetCustom import *

import os
from pathlib import Path


class ImageListWidget(QListWidget):
    def __init__(self, camera_name="None", img_size=640,
                 colors=[(0, 0, 255), (0, 255, 0), (255, 0, 0)], timer_delay=10, parent=None,directory=None,max_items = 20,list_image=[]):

        super().__init__(parent)
        self.camera_name = camera_name
        self.img_size = img_size
        self.colors = colors
        self.max_items= max_items

        self.timer_delay = timer_delay
        self.initUI(camera_name)

        self.list_image = list_image

        # self.timer_watch = QTimer(self)
        # self.timer_watch.timeout.connect(self.update_date_time)
        # self.timer_watch.start(1000)  # Cập nhật mỗi 10ms
        # self.capture = cv2.VideoCapture()
        self.is_warning = False
        self.directory = directory
        self.itemClicked.connect(self.on_item_clicked)

        self.timer_popularlist = QTimer(self)
        self.timer_popularlist.timeout.connect(self.populate_list)
        self.timer_popularlist.start(1000)

        self.label = QLabel(self)


    def populate_list(self):

        for filename in os.listdir(self.directory):
            if filename.lower().endswith(('.jpg', '.jpeg', '.txt')) and filename not in self.list_image:
                self.list_image.append(filename)
                name = os.path.splitext(os.path.basename(filename))[0]
                item = QListWidgetItem(name)
                if self.count() >= 20:
                    # Xóa item đầu tiên (cũ nhất) nếu đã đạt giới hạn
                    self.takeItem(0)
                    # print(f"delete item{self.takeItem(0).text()}")
                self.addItem(item)




    def initUI(self, camera_name="None"):

        # Grid tổng
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(2, 2, 0, 2)

        #self.setGeometry(QtCore.QRect(10, 290, 311, 241))


        # Set chiều rộng, cao cho các grid
        self.layout.setRowMinimumHeight(2, 80)
        self.layout.setRowMinimumHeight(0, 80)

        self.layout.setColumnMinimumWidth(0, 50)
        # self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.setColumnMinimumWidth(2, 50)
        # self.resizeMode(QListWidget.SizeAdjustPolicy)
        self.setLayout(self.layout)

        self.image_folder_path = unidecode(self.camera_name).replace(" ","")
        # creat save image folder
        if not os.path.exists(self.image_folder_path):
            os.makedirs(self.image_folder_path)
        self.frame_count = 0
        self.is_running = False


    def on_item_clicked(self,item):
        # Lấy đường dẫn file từ item đã click
        file_path = "./image_encode/" + item.text()+".jpg"
        #print(item.text())
        self.open_image(file_path)

    def open_image(self,file_path):
        # image_path_decode = "./image_decode/" + self.image_folder_path + datetime.now().strftime(
        #     "%Y%m%d%H%M%S") + "_warning.jpg"
        try:
            with open(file_path, 'r') as text_file:
                image_data = text_file.read()
                # print(encoded_string)
            image = Image.open(BytesIO(image_data))
            image.show()
        except Exception as e:
            print(f"{e}")

