import sys
import cv2
import threading
import queue
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QGridLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal

 # Set device

# VideoCapture with queue to reduce delay
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # Discard previous frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def release(self):
        self.cap.release()

# Thread for processing video
class CameraThread(QThread):
    frame_captured = pyqtSignal(QImage)

    def __init__(self, camera_id):
        super().__init__()
        self.camera_id = camera_id
        self.running = True
        self.video_capture = VideoCapture(camera_id)

    def run(self):
        while self.running:
            frame = self.video_capture.read()
            if frame is not None:
                results = model(frame)
                annotated_frame = results.render()[0]

                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

                self.frame_captured.emit(qt_image)

    def stop(self):
        self.running = False
        self.video_capture.release()
        self.wait()

# Widget for displaying a camera
class CameraWidget(QWidget):
    def __init__(self, camera_id):
        super().__init__()
        self.camera_id = camera_id
        self.label = QLabel(self)
        self.label.setText("Loading...")
        self.label.setStyleSheet("QLabel { background-color: black; color: white; }")
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        
        # Start Camera Thread
        self.camera_thread = CameraThread(camera_id)
        self.camera_thread.frame_captured.connect(self.update_frame)
        self.camera_thread.start()

    def update_frame(self, qt_image):
        self.label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.camera_thread.stop()
        super().closeEvent(event)


