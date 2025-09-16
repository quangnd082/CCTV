import sys, time, queue, cv2, numpy as np
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap

YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

def to_qimage(rgb: np.ndarray) -> QImage:
    """Convert numpy RGB (H,W,3) -> QImage (no copy of data layout change)."""
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

# =========================================
# Capture Thread
# =========================================
class CaptureThread(QThread):
    def __init__(self, out_queue: queue.Queue, src=0, fps_limit=None):
        super().__init__()
        self.out_q = out_queue
        self.src = src
        self.running = True
        self.fps_limit = fps_limit  # e.g., 30 -> sleep to cap FPS

    def run(self):
        cap = cv2.VideoCapture(self.src)
        # Mẹo: tăng buffer camera thấp để giảm trễ
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        last = time.time()
        while self.running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue

            # Throttle nếu cần
            if self.fps_limit:
                now = time.time()
                min_dt = 1.0 / self.fps_limit
                if now - last < min_dt:
                    time.sleep(min_dt - (now - last))
                last = time.time()

            # Đẩy frame mới nhất, nếu full -> drop frame cũ để tránh backlog
            try:
                if self.out_q.full():
                    _ = self.out_q.get_nowait()
                self.out_q.put_nowait(frame)
            except queue.Full:
                pass

        cap.release()

    def stop(self):
        self.running = False
        self.wait(1000)

# =========================================
# Detect Thread
# =========================================
class DetectThread(QThread):
    # Có thể emit FPS để debug
    fps_signal = pyqtSignal(float)

    def __init__(self, in_queue: queue.Queue, out_queue: queue.Queue, model_path="yolov8n.pt", conf=0.25, use_gpu=True):
        super().__init__()
        self.in_q = in_queue
        self.out_q = out_queue
        self.running = True
        self.conf = conf
        self.model = None
        self.use_gpu = use_gpu and YOLO_AVAILABLE
        self.model_path = model_path

    def setup_model(self):
        if YOLO_AVAILABLE:
            self.model = YOLO(self.model_path)
            # Warmup (tùy chọn)
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, conf=self.conf, verbose=False, imgsz=640, device=0 if self.use_gpu else "cpu")

    def draw_boxes_yolo(self, frame_bgr, results):
        # Vẽ bbox đơn giản từ YOLO results
        if not results:
            return frame_bgr
        res = results[0]
        if res.boxes is None:
            return frame_bgr

        boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, "xyxy") else []
        confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, "conf") else []
        clss = res.boxes.cls.cpu().numpy() if hasattr(res.boxes, "cls") else []

        for (x1, y1, x2, y2), cf, c in zip(boxes, confs, clss):
            p1 = (int(x1), int(y1)); p2 = (int(x2), int(y2))
            cv2.rectangle(frame_bgr, p1, p2, (0, 255, 0), 2)
            label = f"{int(c)}:{cf:.2f}"
            cv2.putText(frame_bgr, label, (p1[0], max(0, p1[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return frame_bgr

    def run(self):
        t0 = time.time(); frame_count = 0
        self.setup_model()

        while self.running:
            try:
                frame_bgr = self.in_q.get(timeout=0.05)  # đợi ngắn
            except queue.Empty:
                continue

            start = time.time()
            # Detect
            if self.model is not None:
                results = self.model.predict(frame_bgr, conf=self.conf, verbose=False, imgsz=640, device=0 if self.use_gpu else "cpu")
                vis_bgr = self.draw_boxes_yolo(frame_bgr.copy(), results)
            else:
                # Fallback: không detect, chỉ passthrough + overlay chữ
                vis_bgr = frame_bgr.copy()
                cv2.putText(vis_bgr, "YOLO not installed - passthrough", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

            dt = time.time() - start
            frame_count += 1
            if frame_count % 10 == 0 and dt > 0:
                self.fps_signal.emit(1.0 / dt)

            # Đẩy ảnh đã vẽ sang UI queue, ưu tiên mới nhất
            try:
                if self.out_q.full():
                    _ = self.out_q.get_nowait()
                self.out_q.put_nowait(vis_bgr)
            except queue.Full:
                pass

    def stop(self):
        self.running = False
        self.wait(1000)

# =========================================
# UI
# =========================================
class MainWindow(QWidget):
    def __init__(self, src=0):
        super().__init__()
        self.setWindowTitle("PyQt5 Object Detection - No Delay")
        self.label = QLabel("Waiting frames...")
        self.label.setAlignment(Qt.AlignCenter)

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setAlignment(Qt.AlignRight)

        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        top = QHBoxLayout()
        top.addWidget(self.btn_start)
        top.addWidget(self.btn_stop)
        top.addStretch()
        top.addWidget(self.fps_label)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.label)
        self.setLayout(layout)

        # Queues
        self.cap2det_q = queue.Queue(maxsize=1)  # chống backlog
        self.det2ui_q = queue.Queue(maxsize=1)

        # Threads
        self.cap_th = CaptureThread(self.cap2det_q, src=src, fps_limit=None)
        self.det_th = DetectThread(self.cap2det_q, self.det2ui_q, model_path="yolov8n.pt", conf=0.25, use_gpu=True)
        self.det_th.fps_signal.connect(self.on_fps)

        # Timer UI để lấy ảnh không block
        self.timer = QTimer(self)
        self.timer.setInterval(15)  # ~66Hz
        self.timer.timeout.connect(self.update_view)

        # Signals
        self.btn_start.clicked.connect(self.start_pipeline)
        self.btn_stop.clicked.connect(self.stop_pipeline)

    def start_pipeline(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        if not self.cap_th.isRunning():
            self.cap_th.start()
        if not self.det_th.isRunning():
            self.det_th.start()
        if not self.timer.isActive():
            self.timer.start()

    def stop_pipeline(self):
        self.timer.stop()
        if self.det_th.isRunning():
            self.det_th.stop()
        if self.cap_th.isRunning():
            self.cap_th.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def update_view(self):
        # Lấy frame đã vẽ từ detect -> UI
        try:
            frame_bgr = self.det2ui_q.get_nowait()
        except queue.Empty:
            return

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = to_qimage(rgb)
        self.label.setPixmap(QPixmap.fromImage(qimg))

    def on_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def closeEvent(self, e):
        self.stop_pipeline()
        super().closeEvent(e)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow(src=0)  # đổi src nếu là file/video RTSP
    w.resize(960, 640)
    w.show()
    sys.exit(app.exec_())
