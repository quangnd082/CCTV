import json
import os
from dataclasses import dataclass, field
from typing import List, Optional


SETTINGS_FILE = "settings.json"
SETTINGS_VERSION = 1

_DETECT_FLAG_KEYS = ("fell", "helmet", "jacket", "fire", "smoke")


@dataclass
class DisplaySettings:
    max_active_cameras: int = 4
    monitor_columns: int = 2
    selected_cameras: List[str] = field(default_factory=list)


@dataclass
class PerformanceSettings:
    display_target_fps: float = 6.0
    swap_delay_ms: int = 800


@dataclass
class AppSettings:
    version: int = SETTINGS_VERSION
    display: DisplaySettings = field(default_factory=DisplaySettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)


def _migrate_selected_from_config(config_info) -> List[str]:
    """Đọc use_camera cũ từ config_data (migration một lần)."""
    selected: List[str] = []
    max_n = max(1, int(getattr(config_info, "max_active_cameras", 4) or 4))
    for ci in getattr(config_info, "camera_infos", []):
        flags = getattr(ci, "enable_flags", None) or {}
        if int(flags.get("use_camera", 0)) == 1:
            selected.append(ci.camera_name)
    selected = selected[:max_n]
    if not selected:
        for ci in getattr(config_info, "camera_infos", [])[:max_n]:
            selected.append(ci.camera_name)
    return selected


def default_settings(config_info=None) -> AppSettings:
    settings = AppSettings()
    if config_info is not None:
        settings.display.max_active_cameras = max(
            1, int(getattr(config_info, "max_active_cameras", 4) or 4)
        )
        settings.display.monitor_columns = max(
            1, min(2, int(getattr(config_info, "max_column", 2) or 2))
        )
        settings.display.selected_cameras = _migrate_selected_from_config(config_info)
    return settings


def _validate_selected(settings: AppSettings, config_info=None) -> None:
    if config_info is None:
        return
    valid_names = {ci.camera_name for ci in config_info.camera_infos}
    max_n = settings.display.max_active_cameras
    cleaned = []
    for name in settings.display.selected_cameras:
        if name in valid_names and name not in cleaned:
            cleaned.append(name)
        if len(cleaned) >= max_n:
            break
    if len(cleaned) < len(settings.display.selected_cameras):
        settings.display.selected_cameras = cleaned
    elif not cleaned and config_info.camera_infos:
        settings.display.selected_cameras = _migrate_selected_from_config(config_info)


def _dict_to_settings(data: dict) -> AppSettings:
    display_raw = data.get("display") or {}
    perf_raw = data.get("performance") or {}
    display = DisplaySettings(
        max_active_cameras=int(display_raw.get("max_active_cameras", 4)),
        monitor_columns=int(display_raw.get("monitor_columns", 2)),
        selected_cameras=list(display_raw.get("selected_cameras") or []),
    )
    performance = PerformanceSettings(
        display_target_fps=float(perf_raw.get("display_target_fps", 6.0)),
        swap_delay_ms=int(perf_raw.get("swap_delay_ms", 800)),
    )
    return AppSettings(
        version=int(data.get("version", SETTINGS_VERSION)),
        display=display,
        performance=performance,
    )


def settings_to_dict(settings: AppSettings) -> dict:
    return {
        "version": settings.version,
        "display": {
            "max_active_cameras": settings.display.max_active_cameras,
            "monitor_columns": settings.display.monitor_columns,
            "selected_cameras": list(settings.display.selected_cameras),
        },
        "performance": {
            "display_target_fps": settings.performance.display_target_fps,
            "swap_delay_ms": settings.performance.swap_delay_ms,
        },
    }


def load_settings(
    file_path: str = SETTINGS_FILE,
    config_info=None,
    *,
    create_if_missing: bool = True,
) -> AppSettings:
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        settings = _dict_to_settings(data)
        _validate_selected(settings, config_info)
        return settings

    settings = default_settings(config_info)
    if create_if_missing:
        save_settings(settings, file_path)
    return settings


def save_settings(settings: AppSettings, file_path: str = SETTINGS_FILE) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(settings_to_dict(settings), f, ensure_ascii=False, indent=4)


def strip_use_camera_from_flags(flags: Optional[dict]) -> dict:
    """Loại use_camera khỏi enable_flags (chỉ giữ rule detect)."""
    if not isinstance(flags, dict):
        return {}
    out = {k: flags[k] for k in _DETECT_FLAG_KEYS if k in flags}
    for k, v in flags.items():
        if k not in out and k != "use_camera":
            out[k] = v
    return out
