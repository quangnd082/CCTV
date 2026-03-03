import threading
import queue
import time
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import torch
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QImage
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from datetime import datetime
from unidecode import unidecode

from Logging import Logger


INVALID_CHARS = r'[^A-Za-z0-9_.-]'


def safe_name(s: str) -> str:
    """Biến chuỗi thành tên file an toàn cho Windows."""
    s = os.path.basename(str(s))
    s = s.replace(':', '_')
    s = re.sub(INVALID_CHARS, '_', s)
    return s[:80]


@dataclass
class CameraState:
    camera_name: str
    camera_slug: str
    save_dir: Path
    last_warn_ts: float = 0.0
    last_warning_state: bool = False


class YoloEngine(QObject):
    """
    Engine YOLO dùng chung cho nhiều camera.

    - Cache model theo đường dẫn weight.
    - Worker thread riêng đọc queue các job inference.
    - Phát tín hiệu result_ready về UI cho từng camera.
    """

    result_ready = pyqtSignal(str, QImage, bool, dict)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._models: Dict[str, YOLO] = {}
        self._device = 0 if torch.cuda.is_available() else "cpu"
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=32)
        self._lock = threading.Lock()
        self._running = True
        self._camera_states: Dict[str, CameraState] = {}

        self._worker = threading.Thread(target=self._worker_loop, name="YoloEngineWorker", daemon=True)
        self._worker.start()

    # --------- public API ----------
    def request_inference(
        self,
        camera_name: str,
        frame,
        model_path: str,
        img_size: int,
        yolo_rate: float,
        roi_check,
        classes,
        colors,
        enable_flags,
        logger: Optional[Logger] = None,
    ) -> None:
        """
        Đẩy một frame vào queue để engine xử lý.
        Nếu queue đầy, drop frame cũ nhất để ưu tiên frame mới.
        """
        if frame is None:
            return

        job = {
            "camera_name": camera_name,
            "frame": frame,
            "model_path": model_path,
            "img_size": img_size,
            "yolo_rate": float(yolo_rate),
            "roi_check": roi_check or [-1, 9999, -1, 9999],
            "classes": classes,
            "colors": colors,
            "enable_flags": enable_flags or {},
            "logger": logger,
        }

        try:
            self._queue.put_nowait(job)
        except queue.Full:
            # drop oldest rồi thử lại
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                # nếu vẫn đầy thì bỏ qua frame này
                if logger:
                    logger.warning("YOLO queue overloaded, dropping frame")

    def stop(self):
        """Dừng worker thread."""
        self._running = False

    # --------- nội bộ ----------
    def _get_model(self, model_path: str, logger: Optional[Logger]) -> YOLO:
        with self._lock:
            if model_path in self._models:
                return self._models[model_path]

            if logger:
                logger.info(f"Loading YOLO model: {model_path}")
            model = YOLO(model_path)

            try:
                model.fuse()
                if hasattr(model, "model") and hasattr(model.model, "fuse"):
                    model.model.fuse()
            except Exception:
                pass

            if torch.cuda.is_available():
                try:
                    model.to("cuda")
                    if hasattr(model, "model") and hasattr(model.model, "half"):
                        model.model.half()
                except Exception:
                    pass

            try:
                model.eval()
            except Exception:
                pass

            self._models[model_path] = model
            return model

    def _get_camera_state(self, camera_name: str) -> CameraState:
        if camera_name not in self._camera_states:
            slug = safe_name(unidecode(camera_name))
            save_dir = Path("./LastDetectionWarning")
            save_dir.mkdir(parents=True, exist_ok=True)
            self._camera_states[camera_name] = CameraState(
                camera_name=camera_name,
                camera_slug=slug,
                save_dir=save_dir,
            )
        return self._camera_states[camera_name]

    def _worker_loop(self):
        torch.set_grad_enabled(False)
        while self._running:
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if job is None:
                continue

            try:
                self._process_job(job)
            except Exception as e:
                logger = job.get("logger")
                if logger:
                    logger.error(f"YoloEngine job error: {e}")

    def _process_job(self, job: Dict[str, Any]) -> None:
        camera_name = job["camera_name"]
        frame = job["frame"]
        model_path = job["model_path"]
        img_size = job["img_size"]
        yolo_rate = job["yolo_rate"]
        roi_check = job["roi_check"]
        classes = job["classes"]
        colors = job["colors"]
        enable_flags = job["enable_flags"]
        logger: Optional[Logger] = job.get("logger")

        state = self._get_camera_state(camera_name)
        model = self._get_model(model_path, logger)

        x1, x2, y1, y2 = roi_check

        predict_params = {
            "imgsz": img_size,
            "conf": float(yolo_rate),
            "device": self._device,
            "verbose": False,
            "agnostic_nms": True,
        }
        if torch.cuda.is_available():
            predict_params["half"] = True

        results = model.predict(frame, **predict_params)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotator = Annotator(cv2image, line_width=2, font_size=16)

        is_warning = False
        r = results[0]
        boxes = r.boxes

        for box in boxes:
            try:
                if float(box.conf) <= float(yolo_rate):
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
            helmet_ok = ci == 5 and enable_flags.get("helmet", False)
            fell_ok = ci == 9 and enable_flags.get("fell", False)
            jacket_ok = ci == 8 and enable_flags.get("jacket", False)

            try:
                name_from_cfg = classes[ci] if 0 <= ci < len(classes) else None
            except Exception:
                name_from_cfg = None

            name_from_model = None
            try:
                if hasattr(model, "names") and model.names is not None:
                    if isinstance(model.names, dict):
                        name_from_model = model.names.get(ci)
                    elif 0 <= ci < len(model.names):
                        name_from_model = model.names[ci]
            except Exception:
                pass

            obj_name = (
                name_from_cfg
                if (isinstance(name_from_cfg, str) and len(name_from_cfg) > 0)
                else (name_from_model if name_from_model else str(ci))
            )

            try:
                conf_val = float(box.conf)
            except Exception:
                conf_val = 0.0

            label = f"{obj_name} {conf_val:.2f}"
            color_warn = tuple(colors[ci]) if 0 <= ci < len(colors) else (255, 0, 0)

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
            if now_ts - state.last_warn_ts > 5.0:
                try:
                    ts = datetime.now().strftime("%Y%m%d%H%M%S")
                    file_path = state.save_dir / f"{state.camera_slug}_{ts}_warning.jpg"
                    threading.Thread(
                        target=self._save_image,
                        args=(str(file_path), img.copy(), logger),
                        daemon=True,
                    ).start()
                    state.last_warn_ts = time.time()
                except Exception as e:
                    if logger:
                        logger.error(f"save image start error: {e}")

        if is_warning != state.last_warning_state:
            state.last_warning_state = is_warning
            if is_warning:
                if logger:
                    logger.warning("Violation detected")

        q_image = QImage(img.data, w, h, w * 3, QImage.Format_RGB888).copy()
        meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.result_ready.emit(camera_name, q_image, is_warning, meta)

    @staticmethod
    def _save_image(file_path: str, img_rgb, logger: Optional[Logger]):
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, img_bgr)
        except Exception as e:
            if logger:
                logger.error(f"save image error: {e}")


_ENGINE_INSTANCE: Optional[YoloEngine] = None


def get_yolo_engine() -> YoloEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = YoloEngine()
    return _ENGINE_INSTANCE

