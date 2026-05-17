import time
from typing import List

from Logging import Logger
from Program import MainWindow


def log_multi_camera_status(window: MainWindow, logger: Logger, interval_sec: float = 5.0, duration_sec: float = 60.0):
    """
    Helper đơn giản để hỗ trợ test scale 8–16 camera:
    - Gọi định kỳ để log thời gian, số camera widget, và trạng thái warning hiện tại.
    - Dùng để quan sát khi chạy nhiều camera xem app còn mượt và warning hoạt động ổn.
    """
    start = time.time()
    while time.time() - start < duration_sec:
        try:
            cam_widgets: List = []
            layout = getattr(window, "layout_cam", None)
            if layout is not None:
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    w = item.widget()
                    if w is not None:
                        cam_widgets.append(w)

            active_warnings = sum(getattr(w, "is_warning", False) for w in cam_widgets)
            logger.info(
                f"[SCALE-TEST] cams={len(cam_widgets)}, active_warnings={active_warnings}"
            )
        except Exception as e:
            logger.error(f"[SCALE-TEST] error: {e}")
        time.sleep(interval_sec)

