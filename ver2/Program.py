import sys

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QGridLayout,
    QLabel,
    QTabWidget,
)

from CameraWidget import CameraWidget
from camera_settings_panel import CameraSettingsPanel
from display_image import LogWidget
from config_data import load_config_file
from config_data_log import load_config_log_file
from list_widget import ImageListWidget
from Logging import Logger
from ffmpeg_capture import FFmpegCapture
from yolo_engine import log_yolo_device_status, get_yolo_engine
from settings_store import load_settings, save_settings, SETTINGS_FILE, AppSettings


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_info = None
        self.config_info_log = None
        self.app_settings: AppSettings = None
        self.monitor_slots = []
        self._cam_info_by_name = {}
        self._applied_selection = []
        self.max_active_cameras = 4
        self.monitor_columns = 2
        self._settings_tab_index = -1
        self._ui_ready = False

        self.setWindowTitle("EHS CCTV Monitoring")
        self.setGeometry(100, 100, 1280, 820)

        self.logger = Logger("CCTV")
        FFmpegCapture.log_system_diagnostics(self.logger)
        log_yolo_device_status(self.logger)
        get_yolo_engine()

        self.config_info = load_config_file("config_data.json")
        self.app_settings = load_settings(SETTINGS_FILE, self.config_info)

        self.max_active_cameras = max(
            1, int(self.app_settings.display.max_active_cameras or 4)
        )
        self.monitor_columns = max(
            1, min(2, int(self.app_settings.display.monitor_columns or 2))
        )

        for ci in self.config_info.camera_infos:
            self._cam_info_by_name[ci.camera_name] = ci

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self._build_monitor_tab()
        self._build_settings_tab()
        self.create_log_widgets()

        self.tabs.addTab(self.monitor_tab, "Monitor")
        self._settings_tab_index = self.tabs.addTab(self.settings_panel, "Settings")
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _build_monitor_tab(self) -> None:
        self.monitor_tab = QWidget()
        outer = QGridLayout(self.monitor_tab)
        outer.setContentsMargins(4, 4, 4, 4)

        self.label_active_streams = QLabel()
        self.label_active_streams.setStyleSheet(
            "color: #c8e6ff; font-size: 13px; font-weight: bold; padding: 4px 8px;"
        )
        outer.addWidget(self.label_active_streams, 0, 0, 1, self.monitor_columns)

        cam_grid_host = QWidget()
        self.layout_cam = QGridLayout(cam_grid_host)
        self.layout_cam.setSpacing(4)

        for i in range(self.max_active_cameras):
            slot = CameraWidget(
                camera_name=f"Ô {i + 1}",
                camera_src="",
                stream_enabled=False,
                logger=self.logger,
            )
            self.monitor_slots.append(slot)
            row, col = divmod(i, self.monitor_columns)
            self.layout_cam.addWidget(slot, row, col)

        outer.addWidget(cam_grid_host, 1, 0, 1, self.monitor_columns)

        self.layout_log_host = QWidget()
        self.layout_log = QGridLayout(self.layout_log_host)
        outer.addWidget(self.layout_log_host, 0, self.monitor_columns, 2, 1)

    def _build_settings_tab(self) -> None:
        self.settings_panel = CameraSettingsPanel(
            self.config_info.camera_infos,
            max_active=self.max_active_cameras,
            logger=self.logger,
        )
        self.settings_panel.load_selection(self.app_settings.display.selected_cameras)
        self.settings_panel.save_requested.connect(self._on_save_selection)
        self.settings_panel.apply_requested.connect(self._on_apply_selection)

    def _on_tab_changed(self, index: int) -> None:
        if not self._ui_ready or index != self._settings_tab_index:
            return
        if self.logger:
            self.logger.info("[CAM] entered Settings — stopping all streams")
        self.settings_panel.load_selection(self._applied_selection)
        self._stop_all_streams()

    def _stop_all_streams(self) -> None:
        for slot in self.monitor_slots:
            slot.set_stream_enabled(False)
        self.label_active_streams.setText(
            "Đã dừng tất cả luồng — chọn camera và bấm «Áp dụng và về Monitor»"
        )

    def _sync_settings_from_selection(self, selected_names: list) -> None:
        self.app_settings.display.selected_cameras = list(selected_names)
        self.app_settings.display.max_active_cameras = self.max_active_cameras
        self.app_settings.display.monitor_columns = self.monitor_columns

    def _save_settings_to_file(self, selected_names: list) -> None:
        self._sync_settings_from_selection(selected_names)
        try:
            save_settings(self.app_settings, SETTINGS_FILE)
            if self.logger:
                self.logger.info(f"[SETTINGS] saved to {SETTINGS_FILE}")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[SETTINGS] could not save: {e}")

    def _on_save_selection(self, selected_names: list) -> None:
        self._save_settings_to_file(selected_names)

    def _on_apply_selection(self, selected_names: list) -> None:
        self._save_settings_to_file(selected_names)
        self._apply_selection(selected_names)
        self.tabs.setCurrentIndex(0)

    def _apply_selection(self, selected_names: list) -> None:
        selected_names = list(selected_names or [])[: self.max_active_cameras]
        self._applied_selection = list(selected_names)

        target_fps = float(self.app_settings.performance.display_target_fps)

        for i in range(self.max_active_cameras):
            if i < len(selected_names):
                name = selected_names[i]
                ci = self._cam_info_by_name.get(name)
                if ci is not None:
                    self.monitor_slots[i].bind_from_camera_info(ci, stream_enabled=True)
                    if hasattr(self.monitor_slots[i], "set_target_fps"):
                        self.monitor_slots[i].set_target_fps(target_fps)
                else:
                    if self.logger:
                        self.logger.warning(f"[CAM] unknown camera in settings: {name}")
                    self.monitor_slots[i].show_empty_slot(f"Ô {i + 1}")
            else:
                self.monitor_slots[i].show_empty_slot(f"Ô {i + 1}")

        n = len(selected_names)
        self.label_active_streams.setText(
            f"Monitor: {n}/{self.max_active_cameras} luồng đang chạy"
        )
        if self.logger:
            self.logger.info(f"[CAM] streams started: {selected_names}")

    def create_log_widgets(self) -> None:
        self.config_info_log = load_config_log_file("config_data_log.json")
        camera_name_image = self.config_info_log.log_infos[0].camera_name
        img_size = self.config_info_log.log_infos[0].img_size
        colors = tuple(map(tuple, self.config_info_log.log_infos[0].colors))
        directory = self.config_info_log.log_infos[0].directory

        self.display_image = LogWidget(
            camera_name=camera_name_image,
            img_size=img_size / 2,
            colors=colors,
            directory=directory,
        )
        self.display_image.setMaximumWidth(400)
        self.display_image.setMaximumHeight(400)
        self.layout_log.addWidget(self.display_image, 0, 0, 1, 1)

        camera_name_list = self.config_info_log.log_infos[0].camera_name
        colors = tuple(map(tuple, self.config_info_log.log_infos[0].colors))
        timer_delay = self.config_info_log.log_infos[0].timer_delay
        directory = self.config_info_log.log_infos[0].directory

        self.widget_list = ImageListWidget(
            camera_name=camera_name_list,
            timer_delay=timer_delay,
            colors=colors,
            directory=directory,
        )
        self.widget_list.setMaximumWidth(400)
        self.display_image.setMaximumHeight(600)
        self.layout_log.addWidget(self.widget_list, 1, 0)

        self.settings_panel.load_selection(self.app_settings.display.selected_cameras)
        self._apply_selection(self.app_settings.display.selected_cameras)
        self._ui_ready = True


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
