import json
import subprocess
import threading
import queue
import time
import shutil
from typing import Optional, Tuple

import cv2
import numpy as np


def _read_exact(stream, nbytes: int) -> bytes:
    """Đọc đủ nbytes từ pipe (tránh frame raw bị cắt đôi)."""
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = stream.read(nbytes - len(buf))
        if not chunk:
            raise RuntimeError("ffmpeg stdout ended (short read)")
        buf.extend(chunk)
    return bytes(buf)


def _even(n: int) -> int:
    return max(2, int(n) // 2 * 2)


class FFmpegCapture:
    """
    Production RTSP ingest qua FFmpeg subprocess.

    Pipeline mặc định (raw, không MJPEG):
        RTSP/TCP -> [NVDEC nếu có] -> H264 decode -> scale -> BGR24 raw pipe -> NumPy

    So với MJPEG pipe cũ, bỏ được:
        - MJPEG encode (FFmpeg)
        - JPEG parse + cv2.imdecode (Python)

    Giữ nguyên: TCP, nobuffer, queue maxsize=1, watchdog stall, exponential backoff.
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
        use_hwaccel: bool = True,
        prefer_raw: bool = True,
        logger=None,
    ):
        self.source = source
        self.target_width = target_width
        self.rtsp_transport = rtsp_transport
        self.open_timeout_sec = float(open_timeout_sec)
        self.read_timeout_sec = float(read_timeout_sec)
        self.reconnect_delay_sec = float(reconnect_delay_sec)
        self.reconnect_delay_max_sec = float(reconnect_delay_max_sec)
        self._use_hwaccel_requested = bool(use_hwaccel)
        self._cuda_ok = bool(use_hwaccel) and self._cuda_hwaccel_available()
        self._scale_cuda_ok = self._cuda_ok and self._filter_available("scale_cuda")
        self._use_hwaccel = self._cuda_ok
        self._hwaccel_failed = False
        self._cuvid_decoder: Optional[str] = None
        self._video_codec = ""
        self.decode_backend = "cpu"
        self.prefer_raw = bool(prefer_raw)
        self.logger = logger

        self._proc: Optional[subprocess.Popen] = None
        self._running = True
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self._last_frame_ok_ts = 0.0
        self._stall_timeout_sec = 5.0

        self._out_w = 0
        self._out_h = 0
        self._frame_bytes = 0
        self._output_mode = "raw"  # "raw" | "mjpeg"

        self._resolve_output_geometry()

        self._t = threading.Thread(target=self._reader_loop, name="FFmpegCaptureReader", daemon=True)
        self._start_ffmpeg()
        self._t.start()

    @staticmethod
    def is_available() -> bool:
        return shutil.which("ffmpeg") is not None

    @staticmethod
    def _ffprobe_available() -> bool:
        return shutil.which("ffprobe") is not None

    @staticmethod
    def _ffmpeg_run(args: list, timeout: float = 5.0) -> subprocess.CompletedProcess:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    @staticmethod
    def _cuda_hwaccel_available() -> bool:
        if not FFmpegCapture.is_available():
            return False
        try:
            r = FFmpegCapture._ffmpeg_run(["ffmpeg", "-hide_banner", "-hwaccels"])
            return "cuda" in (r.stdout or "").lower()
        except Exception:
            return False

    @staticmethod
    def _filter_available(name: str) -> bool:
        if not FFmpegCapture.is_available():
            return False
        try:
            r = FFmpegCapture._ffmpeg_run(["ffmpeg", "-hide_banner", "-filters"])
            return name in (r.stdout or "")
        except Exception:
            return False

    @staticmethod
    def _decoder_available(name: str) -> bool:
        if not FFmpegCapture.is_available():
            return False
        try:
            r = FFmpegCapture._ffmpeg_run(["ffmpeg", "-hide_banner", "-decoders"])
            return name in (r.stdout or "")
        except Exception:
            return False

    @staticmethod
    def _cuvid_decoder_for_codec(codec: str) -> Optional[str]:
        c = (codec or "").lower().strip()
        if c in ("h264", "avc"):
            return "h264_cuvid" if FFmpegCapture._decoder_available("h264_cuvid") else None
        if c in ("hevc", "h265"):
            return "hevc_cuvid" if FFmpegCapture._decoder_available("hevc_cuvid") else None
        return None

    @staticmethod
    def log_system_diagnostics(logger=None) -> dict:
        """
        Ghi một lần lúc mở app — dùng đối chiếu bảng log trong README.
        Trả về dict capability để test/script có thể đọc.
        """
        caps = {
            "ffmpeg": FFmpegCapture.is_available(),
            "ffprobe": FFmpegCapture._ffprobe_available(),
            "cuda_hwaccel": FFmpegCapture._cuda_hwaccel_available(),
            "h264_cuvid": FFmpegCapture._decoder_available("h264_cuvid"),
            "hevc_cuvid": FFmpegCapture._decoder_available("hevc_cuvid"),
            "scale_cuda": FFmpegCapture._filter_available("scale_cuda"),
        }

        def _emit(level: str, msg: str) -> None:
            print(msg)
            if logger is not None:
                getattr(logger, level)(msg)

        _emit("info", "[INGEST] === FFmpeg / NVDEC system check ===")
        for key, ok in caps.items():
            _emit("info", f"[INGEST] {key}: {'YES' if ok else 'NO'}")

        if not caps["ffmpeg"]:
            _emit("warning", "[INGEST] WARN ffmpeg missing in PATH -> RTSP will use OpenCV fallback")
        elif not caps["ffprobe"]:
            _emit("warning", "[INGEST] WARN ffprobe missing -> per-camera MJPEG fallback likely")
        elif caps["cuda_hwaccel"] and caps["h264_cuvid"]:
            _emit("info", "[INGEST] OK NVDEC-ready (install full FFmpeg build with CUDA)")
        elif caps["cuda_hwaccel"]:
            _emit("warning", "[INGEST] WARN cuda yes but no h264_cuvid -> decode may stay on CPU")
        else:
            _emit(
                "warning",
                "[INGEST] WARN no CUDA in ffmpeg -> ingest CPU only (use gyan.dev FULL build)",
            )
        return caps

    def _scale_mode(self) -> str:
        if self._output_mode != "raw":
            return "n/a"
        if self._use_hwaccel and not self._hwaccel_failed and self._scale_cuda_ok:
            return "scale_cuda"
        return "scale_cpu"

    @property
    def pipeline_summary(self) -> str:
        """Chuỗi ngắn để grep trong Log_Vision/Log_View/*.log"""
        if self._output_mode == "mjpeg":
            dec = self.decode_backend if not self._hwaccel_failed else "cpu"
            return f"mjpeg_fallback|decode={dec}"
        return (
            f"raw {self._out_w}x{self._out_h}|decode={self.decode_backend}|{self._scale_mode()}"
        )

    def _stimeout_us(self) -> int:
        return int(self.open_timeout_sec * 1_000_000)

    def _rw_timeout_us(self) -> int:
        return int(self.read_timeout_sec * 1_000_000)

    def _probe_stream_info(self) -> Tuple[int, int, str]:
        """Lấy width/height/codec bằng ffprobe (không decode full stream)."""
        if not self._ffprobe_available():
            raise RuntimeError("ffprobe not found")

        cmd = [
            "ffprobe",
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            self.rtsp_transport,
            "-stimeout",
            str(self._stimeout_us()),
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name",
            "-of",
            "json",
            self.source,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=max(8.0, self.open_timeout_sec + 3.0))
        if r.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {(r.stderr or r.stdout or '').strip()}")

        data = json.loads(r.stdout or "{}")
        streams = data.get("streams") or []
        if not streams:
            raise RuntimeError("ffprobe: no video stream")

        s0 = streams[0]
        w, h = int(s0["width"]), int(s0["height"])
        codec = str(s0.get("codec_name") or "")
        if w <= 0 or h <= 0:
            raise RuntimeError(f"invalid probe size: {w}x{h}")
        return w, h, codec

    def _configure_hwaccel(self, codec: str) -> None:
        self._video_codec = codec
        if not self._cuda_ok or self._hwaccel_failed:
            self._use_hwaccel = False
            self._cuvid_decoder = None
            self.decode_backend = "cpu"
            return

        self._cuvid_decoder = self._cuvid_decoder_for_codec(codec)
        self._use_hwaccel = True
        if self._cuvid_decoder:
            self.decode_backend = f"nvdec ({self._cuvid_decoder})"
        else:
            self.decode_backend = "cuda"

    def _compute_output_size(self, native_w: int, native_h: int) -> Tuple[int, int]:
        tw = self.target_width
        if isinstance(tw, int) and tw > 0 and native_w > 0 and native_h > 0:
            out_w = _even(min(tw, native_w))
            out_h = _even(int(round(native_h * (out_w / float(native_w)))))
            return out_w, out_h
        return _even(native_w), _even(native_h)

    def _resolve_output_geometry(self) -> None:
        if self.prefer_raw and self._ffprobe_available():
            try:
                nw, nh, codec = self._probe_stream_info()
                self._configure_hwaccel(codec)
                self._out_w, self._out_h = self._compute_output_size(nw, nh)
                self._frame_bytes = self._out_w * self._out_h * 3
                self._output_mode = "raw"
                if self.logger:
                    self.logger.info(
                        f"[INGEST] OK | {self.pipeline_summary} | native {nw}x{nh} codec={codec or '?'}"
                    )
                return
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[INGEST] WARN probe failed, MJPEG fallback: {e}")

        self._output_mode = "mjpeg"
        self._out_w = 0
        self._out_h = 0
        self._frame_bytes = 0
        if self.logger:
            self.logger.warning("[INGEST] WARN MJPEG fallback mode (higher CPU than raw BGR)")

    def _input_args(self) -> list:
        return [
            "-rtsp_transport",
            self.rtsp_transport,
            "-stimeout",
            str(self._stimeout_us()),
            "-rw_timeout",
            str(self._rw_timeout_us()),
            "-probesize",
            "32768",
            "-analyzeduration",
            "0",
            "-i",
            self.source,
        ]

    def _hwaccel_input_prefix(self) -> list:
        if not self._use_hwaccel or self._hwaccel_failed:
            return []
        args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        if self._cuvid_decoder:
            args += ["-c:v", self._cuvid_decoder]
        return args

    def _video_filter(self) -> Optional[str]:
        parts = []
        use_gpu_scale = (
            self._use_hwaccel
            and not self._hwaccel_failed
            and self._scale_cuda_ok
            and self._output_mode == "raw"
            and self._out_w > 0
            and self._out_h > 0
        )

        if use_gpu_scale:
            parts.append(f"scale_cuda={self._out_w}:{self._out_h}")
            parts.append("hwdownload")
            parts.append("format=bgr24")
        elif self._use_hwaccel and not self._hwaccel_failed:
            parts.append("hwdownload")
            parts.append("format=nv12")
            if self._output_mode == "raw" and self._out_w > 0 and self._out_h > 0:
                parts.append(f"scale={self._out_w}:{self._out_h}")
        elif self._output_mode == "raw" and self._out_w > 0 and self._out_h > 0:
            parts.append(f"scale={self._out_w}:{self._out_h}")
        elif self._output_mode == "mjpeg" and isinstance(self.target_width, int) and self.target_width > 0:
            parts.append(f"scale={self.target_width}:-2")

        if not parts:
            return None
        return ",".join(parts)

    def _build_cmd(self) -> list:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
        ]
        cmd += self._hwaccel_input_prefix()
        cmd += self._input_args()
        cmd += [
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-an",
        ]

        vf = self._video_filter()
        if vf:
            cmd += ["-vf", vf]

        if self._output_mode == "raw":
            cmd += [
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "-",
            ]
        else:
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
            mode = self._output_mode
            extra = f" {self._out_w}x{self._out_h}" if mode == "raw" else ""
            self.logger.info(f"[INGEST] start {mode}{extra} | {self.pipeline_summary} | {self.source}")
        try:
            self._proc = subprocess.Popen(
                self._build_cmd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except Exception as e:
            self._proc = None
            if self.logger:
                self.logger.error(f"[INGEST] ERROR spawn failed: {e}")

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

    def _restart(self, delay: float, disable_hwaccel: bool = False) -> None:
        if disable_hwaccel and self._use_hwaccel:
            self._hwaccel_failed = True
            self._use_hwaccel = False
            if self.logger:
                self.logger.warning("[INGEST] WARN CUDA/NVDEC disabled after failure -> CPU decode")
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

    def _read_raw_frame(self, proc: subprocess.Popen) -> np.ndarray:
        assert proc.stdout is not None
        raw = _read_exact(proc.stdout, self._frame_bytes)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self._out_h, self._out_w, 3))
        return np.ascontiguousarray(frame)

    def _read_mjpeg_frames(self, proc: subprocess.Popen, buf: bytearray) -> bytearray:
        assert proc.stdout is not None
        soi = b"\xff\xd8"
        eoi = b"\xff\xd9"

        chunk = proc.stdout.read(65536)
        if not chunk:
            raise RuntimeError("ffmpeg stdout ended")
        buf += chunk

        while True:
            start = buf.find(soi)
            if start < 0:
                if len(buf) > 2_000_000:
                    del buf[:-1_000_000]
                break
            end = buf.find(eoi, start + 2)
            if end < 0:
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
            break

        return buf

    def _reader_loop(self) -> None:
        backoff = float(self.reconnect_delay_sec)
        mjpeg_buf = bytearray()
        raw_failures = 0

        while self._running:
            if self._last_frame_ok_ts > 0 and (time.monotonic() - self._last_frame_ok_ts) > float(
                self._stall_timeout_sec
            ):
                if self.logger:
                    self.logger.warning("[INGEST] WARN watchdog stalled stream, restarting...")
                self._restart(backoff)
                backoff = min(backoff * 2.0, float(self.reconnect_delay_max_sec))
                mjpeg_buf.clear()
                self._last_frame_ok_ts = time.monotonic()

            proc = self._proc
            if proc is None or proc.stdout is None:
                if not self.is_available():
                    if self.logger:
                        self.logger.warning("[INGEST] ERROR ffmpeg not found, stop reader loop")
                    break
                if self.logger:
                    self.logger.warning("[INGEST] WARN ffmpeg process died, restarting...")
                self._restart(backoff)
                backoff = min(backoff * 2.0, float(self.reconnect_delay_max_sec))
                continue

            try:
                if self._output_mode == "raw":
                    frame = self._read_raw_frame(proc)
                    self._put_frame(frame)
                    self._last_frame_ok_ts = time.monotonic()
                    backoff = float(self.reconnect_delay_sec)
                    raw_failures = 0
                else:
                    before = self._last_frame_ok_ts
                    mjpeg_buf = self._read_mjpeg_frames(proc, mjpeg_buf)
                    if self._last_frame_ok_ts != before:
                        backoff = float(self.reconnect_delay_sec)

            except Exception as e:
                raw_failures += 1
                disable_hw = (
                    self._output_mode == "raw"
                    and self._use_hwaccel
                    and not self._hwaccel_failed
                    and raw_failures <= 2
                )

                if self._output_mode == "raw" and raw_failures >= 3 and self.prefer_raw:
                    if self.logger:
                        self.logger.warning(
                            f"[INGEST] WARN raw unstable ({e}), switching to MJPEG fallback"
                        )
                    self._output_mode = "mjpeg"
                    self._frame_bytes = 0
                    raw_failures = 0
                    disable_hw = False

                if self.logger:
                    self.logger.warning(f"[INGEST] WARN reconnect: {e}")
                self._restart(backoff, disable_hwaccel=disable_hw)
                backoff = min(backoff * 2.0, float(self.reconnect_delay_max_sec))
                mjpeg_buf.clear()

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
