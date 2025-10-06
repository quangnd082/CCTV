import json


class CameraInfo:
    def __init__(self, camera_name, camera_src,
                 img_size, yolo_model_path,
                 yolo_rate, classes, colors, is_fell_check,is_helmet_check,is_jacket_check,is_smoke_check,
                 is_fire_check,
                 timer_delay, roi_check):
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
        return cls(camera_info['camera_name'], camera_info['camera_src'], camera_info['img_size'],
                   camera_info['yolo_model_path'], camera_info['yolo_rate'], camera_info['classes'],
                   camera_info['colors'], camera_info['is_fell_check'], camera_info['is_helmet_check'],
                   camera_info['is_jacket_check'],camera_info['is_fire_check'], camera_info['is_smoke_check'],
                    camera_info['timer_delay'], camera_info['roi_check'])


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
