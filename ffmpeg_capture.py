import subprocess
import threading
import queue
import time
import shutil
from typing import Optional, Tuple

import cv2
import numpy as np


class FFmpegCapture:
    """
    Production-ish RTSP capture bằng ffmpeg:
    - RTSP over TCP, timeout, low-latency flags.
    - Output MJPEG qua stdout (image2pipe) để không cần biết trước width/height.
    - Thread reader parse JPEG SOI/EOI, decode thành BGR frame.
    - Queue maxsize=1 (drop frame cũ) để giữ latency thấp.
    - Auto-restart với exponential backoff khi ffmpeg lỗi / stream đứt.
    """

    def __init__(
        self,
        source: str,
        target_width: Optional[int] = None,
        rtsp_transport: str = "tcp",
        open_timeout_sec: float = 5.0,
        read_timeout_sec: float = 5.0,
        reconnect_delay_sec: float = 2.0,
        reconnect_delay_max_sec: float = 30.0,
        logger=None,
    ):
        self.source = source
        self.target_width = target_width
        self.rtsp_transport = rtsp_transport
        self.open_timeout_sec = float(open_timeout_sec)
        self.read_timeout_sec = float(read_timeout_sec)
        self.reconnect_delay_sec = float(reconnect_delay_sec)
        self.reconnect_delay_max_sec = float(reconnect_delay_max_sec)
        self.logger = logger

        self._proc: Optional[subprocess.Popen] = None
        self._running = True
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self._t = threading.Thread(target=self._reader_loop, name="FFmpegCaptureReader", daemon=True)
        self._last_frame_ok_ts = 0.0
        self._stall_timeout_sec = 5.0

        self._start_ffmpeg()
        self._t.start()

    @staticmethod
    def is_available() -> bool:
        return shutil.which("ffmpeg") is not None

    def _build_cmd(self) -> list:
        # ffmpeg timeouts are in microseconds for -stimeout, and in microseconds for -rw_timeout
        stimeout_us = int(self.open_timeout_sec * 1_000_000)
        rw_timeout_us = int(self.read_timeout_sec * 1_000_000)

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            self.rtsp_transport,
            "-stimeout",
            str(stimeout_us),
            "-rw_timeout",
            str(rw_timeout_us),
            "-i",
            self.source,
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-an",
        ]

        if isinstance(self.target_width, int) and self.target_width > 0:
            # giữ aspect ratio
            cmd += ["-vf", f"scale={self.target_width}:-1"]

        # MJPEG stream to stdout
        cmd += [
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-q:v",
            "5",
            "-",
        ]
        return cmd

    def _start_ffmpeg(self) -> None:
        if self.logger:
            self.logger.info(f"FFmpegCapture start: {self.source}")
        try:
            self._proc = subprocess.Popen(
                self._build_cmd(),
                stdout=subprocess.PIPE,
                # tránh deadlock nếu ffmpeg spam log
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except Exception as e:
            self._proc = None
            if self.logger:
                self.logger.error(f"FFmpegCapture spawn error: {e}")

    def _kill_proc(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=1.0)
        except Exception:
            pass

    def _restart(self, delay: float) -> None:
        self._kill_proc()
        time.sleep(max(0.0, float(delay)))
        self._start_ffmpeg()

    def _put_frame(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None:
            return
        if not self._q.empty():
            try:
                self._q.get_nowait()
            except Exception:
                pass
        try:
            self._q.put_nowait(frame_bgr)
        except Exception:
            pass

    def _reader_loop(self) -> None:
        backoff = float(self.reconnect_delay_sec)
        buf = bytearray()
        soi = b"\xff\xd8"  # JPEG start
        eoi = b"\xff\xd9"  # JPEG end

        while self._running:
            # watchdog: nếu lâu không có frame OK thì restart
            if self._last_frame_ok_ts > 0 and (time.monotonic() - self._last_frame_ok_ts) > float(self._stall_timeout_sec):
                if self.logger:
                    self.logger.warning("FFmpegCapture watchdog: stalled stream, restarting...")
                self._restart(backoff)
                backoff = min(backoff * 2.0, float(self.reconnect_delay_max_sec))
                buf.clear()
                self._last_frame_ok_ts = time.monotonic()

            proc = self._proc
            if proc is None or proc.stdout is None:
                if not self.is_available():
                    # ffmpeg không tồn tại; thoát loop để caller fallback
                    if self.logger:
                        self.logger.warning("FFmpegCapture: ffmpeg not found, stop reader loop")
                    break
                if self.logger:
                    self.logger.warning("FFmpegCapture: process not running, restarting...")
                self._restart(backoff)
                backoff = min(backoff * 2.0, float(self.reconnect_delay_max_sec))
                continue

            try:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    raise RuntimeError("ffmpeg stdout ended")
                buf += chunk

                # parse JPEG frames from buffer
                while True:
                    start = buf.find(soi)
                    if start < 0:
                        # giữ buffer nhỏ gọn
                        if len(buf) > 2_000_000:
                            buf = buf[-1_000_000:]
                        break
                    end = buf.find(eoi, start + 2)
                    if end < 0:
                        # chưa đủ frame
                        if start > 0:
                            del buf[:start]
                        break

                    jpg = bytes(buf[start : end + 2])
                    del buf[: end + 2]

                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        self._put_frame(frame)
                        self._last_frame_ok_ts = time.monotonic()
                        backoff = float(self.reconnect_delay_sec)

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"FFmpegCapture reconnect... ({e})")
                self._restart(backoff)
                backoff = min(backoff * 2.0, float(self.reconnect_delay_max_sec))
                buf.clear()

    def read(self, timeout: float = 0.5):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def release(self) -> None:
        self._running = False
        self._kill_proc()
        try:
            if self._t.is_alive():
                self._t.join(timeout=1.0)
        except Exception:
            pass

