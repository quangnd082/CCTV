import json


def _hex_to_rgb_list(s: str):
    s = str(s).strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        raise ValueError(f"Invalid hex color: {s!r}")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return [r, g, b]


def _normalize_color(c):
    """
    Hỗ trợ 2 format:
    - [r,g,b] (int)
    - \"#RRGGBB\" (hex string)
    """
    if isinstance(c, str):
        return _hex_to_rgb_list(c)
    return c


class CameraInfo:
    def __init__(self, camera_name, camera_src,
                 img_size, yolo_model_path,
                 yolo_rate, classes, colors, is_fell_check,is_helmet_check,is_jacket_check,is_smoke_check,
                 is_fire_check,
                 timer_delay, roi_check, enable_flags=None):
        self.camera_name = camera_name
        self.camera_src = camera_src
        self.img_size = img_size
        self.yolo_model_path = yolo_model_path
        self.yolo_rate = yolo_rate
        self.classes = classes
        self.colors = colors
        self.is_fell_check = is_fell_check
        self.is_helmet_check = is_helmet_check
        self.is_jacket_check = is_jacket_check
        self.is_fire_check = is_fire_check
        self.is_smoke_check = is_smoke_check
        self.timer_delay = timer_delay
        # enable_flags có thể chứa cả cờ bật camera và cờ bật/tắt rule detect
        # Ví dụ:
        # {"use_camera": 1, "fell": 1, "helmet": 1, "jacket": 0, "fire": 0, "smoke": 0}
        self.enable_flags = enable_flags or {}

        self.roi_check = roi_check

    def to_dict(self):
        return {
            'camera_name': self.camera_name,
            'camera_src': self.camera_src,
            'img_size': self.img_size,
            'yolo_model_path': self.yolo_model_path,
            'yolo_rate': self.yolo_rate,
            'classes': self.classes,
            'colors': self.colors,
            'enable_flags': self.enable_flags,
            'is_fell_check': self.is_fell_check,
            'is_helmet_check': self.is_helmet_check,
            'is_jacket_check': self.is_jacket_check,
            'is_fire_check': self.is_fire_check,
            'is_smoke_check': self.is_smoke_check,
            'timer_delay': self.timer_delay,
            'roi_check': self.roi_check
        }

    @classmethod
    def from_dict(cls, camera_info):
        enable_flags = camera_info.get("enable_flags")

        # Backward compatible:
        # - Nếu enable_flags không có -> suy ra từ is_*_check và mặc định dùng camera
        # - Nếu enable_flags là int (0/1) -> hiểu là use_camera
        if enable_flags is None:
            enable_flags = {
                "use_camera": 1,
                "fell": camera_info.get("is_fell_check", 0),
                "helmet": camera_info.get("is_helmet_check", 0),
                "jacket": camera_info.get("is_jacket_check", 0),
                "fire": camera_info.get("is_fire_check", 0),
                "smoke": camera_info.get("is_smoke_check", 0),
            }
        elif isinstance(enable_flags, int):
            enable_flags = {
                "use_camera": int(enable_flags),
                "fell": camera_info.get("is_fell_check", 0),
                "helmet": camera_info.get("is_helmet_check", 0),
                "jacket": camera_info.get("is_jacket_check", 0),
                "fire": camera_info.get("is_fire_check", 0),
                "smoke": camera_info.get("is_smoke_check", 0),
            }
        elif isinstance(enable_flags, dict):
            enable_flags.setdefault("use_camera", 1)

        colors = [_normalize_color(c) for c in camera_info.get("colors", [])]
        is_fell_check = camera_info.get("is_fell_check", 0)
        is_helmet_check = camera_info.get("is_helmet_check", 0)
        is_jacket_check = camera_info.get("is_jacket_check", 0)
        is_fire_check = camera_info.get("is_fire_check", 0)
        is_smoke_check = camera_info.get("is_smoke_check", 0)

        return cls(camera_info.get('camera_name'), camera_info.get('camera_src'), camera_info.get('img_size', 640),
                   camera_info.get('yolo_model_path', ''), camera_info.get('yolo_rate', 0.5), camera_info.get('classes', []),
                   colors, is_fell_check, is_helmet_check,
                   is_jacket_check, is_fire_check, is_smoke_check,
                   camera_info.get('timer_delay', 100), camera_info.get('roi_check', [-1, 9999, -1, 9999]), enable_flags=enable_flags)


class ConfigInfo:
    def __init__(self, no_of_camera, max_column, camera_infos):
        self.no_of_camera = no_of_camera
        self.max_column = max_column
        self.camera_infos = camera_infos  # This is an instance of the Address class

    def to_dict(self):
        return {
            'no_of_camera': self.no_of_camera,
            'max_column': self.max_column,
            'camera_infos': [camera_info.to_dict() for camera_info in self.camera_infos]
            # Convert each Address to a dictionary
        }

    @classmethod
    def from_dict(cls, config_info):
        camera_infos = [CameraInfo.from_dict(camera_info) for camera_info in config_info['camera_infos']]
        return cls(config_info['no_of_camera'], config_info['max_column'], camera_infos)



def save_config_file(config_info_class, config_file_path):
    # Convert the Person instance to a dictionary
    config_info_data = config_info_class.to_dict()
    # Write the dictionary to a JSON file
    with open(config_file_path, 'w', encoding="utf-8") as file:
        json.dump(config_info_data, file, ensure_ascii=False, indent=4)


def load_config_file(file_path):
    # Read the JSON file and convert it back to a dictionary
    with open(file_path, 'r', encoding='utf-8') as file:
        config_info_data_read = json.load(file)
    # Create an instance of Person from the dictionary
    config_info_class = ConfigInfo.from_dict(config_info_data_read)
    return config_info_class
