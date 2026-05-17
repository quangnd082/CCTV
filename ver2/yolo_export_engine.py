"""
Export YOLO .pt -> TensorRT .engine trên đúng GPU máy deploy (RTX 4060).

Chạy một lần trên PC production:
    python yolo_export_engine.py --weights yolo11n.pt --batch 4 --imgsz 640

Sau đó trong config_data.json:
    "yolo_model_path": "yolo11n.engine"
"""
from __future__ import annotations

import argparse
import sys

import torch
from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Ultralytics YOLO to TensorRT .engine")
    parser.add_argument("--weights", default="yolo11n.pt", help="Input .pt path")
    parser.add_argument("--imgsz", type=int, default=640, help="Must match config img_size")
    parser.add_argument("--batch", type=int, default=4, help="Max batch in YoloEngine (_batch_size)")
    parser.add_argument("--half", action="store_true", default=True, help="FP16 engine (recommended)")
    parser.add_argument("--dynamic", action="store_true", default=True, help="Dynamic batch 1..N")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[EXPORT] ERROR: CUDA required — build engine on the same GPU as production.")
        return 1

    print(f"[EXPORT] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[EXPORT] weights={args.weights} imgsz={args.imgsz} batch={args.batch} half={args.half}")

    model = YOLO(args.weights)
    out = model.export(
        format="engine",
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic,
        batch=args.batch,
        device=0,
    )
    print(f"[EXPORT] OK -> {out}")
    print("[EXPORT] Update config_data.json: yolo_model_path to the .engine file above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
