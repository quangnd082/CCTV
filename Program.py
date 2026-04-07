import sys
import math
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QPushButton, QVBoxLayout

from CameraWidget import CameraWidget


from display_image import LogWidget

from config_data import *
from config_data_log import *

from list_widget import ImageListWidget
from Logging import Logger

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_info = None
        self.config_info_log = None
        self.setWindowTitle('EHS CCTV Monitoring')
        self.setGeometry(100, 100, 800, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout_cam = QGridLayout()

        self.layout_log = QGridLayout()

        self.layout = QGridLayout(self.central_widget)

        # Logger dùng chung cho toàn app
        self.logger = Logger('CCTV')

        self.create_camera_widgets()
        self.create_log_widgets()

        self.layout.setColumnMinimumWidth(1, 220)
        self.layout.addLayout(self.layout_cam,0,0)
        self.layout.addLayout(self.layout_log, 0, 1)

    def create_camera_widgets(self):
        # load config file
        # try:
        self.config_info = load_config_file("config_data.json")
        # Lọc camera theo enable_flags.use_camera (1: dùng, 0: bỏ qua)
        enabled_camera_infos = []
        for ci in self.config_info.camera_infos:
            flags = getattr(ci, "enable_flags", None) or {}
            use_cam = int(flags.get("use_camera", 1))
            if use_cam == 1:
                enabled_camera_infos.append(ci)

        no_of_camera = len(enabled_camera_infos)
        # Tự động tính số cột theo số camera
        # Ví dụ: 1→1 cột, 2–4→2 cột, 5–6→3 cột
        auto_cols = max(1, math.ceil(math.sqrt(no_of_camera)))
        max_column = auto_cols

        for i, cam_info in enumerate(enabled_camera_infos):
            camera_name = cam_info.camera_name
            camera_src = cam_info.camera_src
            img_size = cam_info.img_size
            yolo_model_path = cam_info.yolo_model_path
            yolo_rate = cam_info.yolo_rate
            classes = cam_info.classes
            colors = tuple(map(tuple, cam_info.colors))

            # Ưu tiên enable_flags (mới), fallback sang is_*_check (cũ)
            flags = getattr(cam_info, "enable_flags", None) or {}
            is_fell_check = bool(int(flags.get("fell", cam_info.is_fell_check)) == 1)
            is_helmet_check = bool(int(flags.get("helmet", cam_info.is_helmet_check)) == 1)
            is_jacket_check = bool(int(flags.get("jacket", cam_info.is_jacket_check)) == 1)
            is_fire_check = bool(int(flags.get("fire", cam_info.is_fire_check)) == 1)
            is_smoke_check = bool(int(flags.get("smoke", cam_info.is_smoke_check)) == 1)
            timer_delay = cam_info.timer_delay

            roi_check = cam_info.roi_check

            widget = CameraWidget(camera_name=camera_name, camera_src=camera_src, img_size=img_size,
                                  yolo_model_path=yolo_model_path,
                                  yolo_rate=yolo_rate, is_fell_check=is_fell_check, is_heltmet_check=is_helmet_check,
                                  is_jacket_check=is_jacket_check,is_fire_check=is_fire_check,is_smoke_check=is_smoke_check,
                                  timer_delay=timer_delay, classes=classes, colors=colors, roi_check=roi_check,logger=self.logger)
            no_of_column = i % max_column
            no_of_row = i // max_column
            self.layout_cam.addWidget(widget, no_of_row, no_of_column)

    def create_log_widgets(self):
        self.config_info_log = load_config_log_file("config_data_log.json")
        camera_name_image = self.config_info_log.log_infos[0].camera_name
        # camera_src = self.config_info_log.log_infos[i].camera_src
        img_size = self.config_info_log.log_infos[0].img_size
        colors = tuple(map(tuple, self.config_info_log.log_infos[0].colors))
        directory = self.config_info_log.log_infos[0].directory

        self.display_image = LogWidget(camera_name=camera_name_image, img_size=img_size / 2,
                                       colors=colors, directory=directory)
        self.display_image.setMaximumWidth(400)
        self.display_image.setMaximumHeight(400)
        self.layout_log.addWidget(self.display_image, 0, 0, 1, 1)

        camera_name_list = self.config_info_log.log_infos[0].camera_name
        colors = tuple(map(tuple, self.config_info_log.log_infos[0].colors))
        timer_delay = self.config_info_log.log_infos[0].timer_delay
        directory = self.config_info_log.log_infos[0].directory

        self.widget_list = ImageListWidget(camera_name=camera_name_list,
                                           timer_delay=timer_delay, colors=colors, directory=directory)
        self.widget_list.setMaximumWidth(400)
        self.display_image.setMaximumHeight(600)
        self.layout_log.addWidget(self.widget_list, 1, 0)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
