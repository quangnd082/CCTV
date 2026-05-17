import json


class LogInfo:
    def __init__(self, camera_name, camera_src,
                 img_size, colors,
                 timer_delay,directory):
        self.camera_name = camera_name
        self.camera_src = camera_src
        self.img_size = img_size
        self.colors = colors
        self.timer_delay = timer_delay
        self.directory = directory


    def to_dict(self):
        return {
            'camera_name': self.camera_name,
            'camera_src': self.camera_src,
            'img_size': self.img_size,
            'colors': self.colors,
            'timer_delay': self.timer_delay,
            'directory' : self.directory
        }

    @classmethod
    def from_dict(cls, log_info):
        return cls(log_info['camera_name'], log_info['camera_src'], log_info['img_size'],
                   log_info['colors'], log_info['timer_delay'],log_info['directory'])




class ConfigInfo:
    def __init__(self, no_of_camera, max_column, log_infos):
        self.no_of_camera = no_of_camera
        self.max_column = max_column
        self.log_infos = log_infos  # This is an instance of the Address class

    def to_dict(self):
        return {
            'no_of_camera': self.no_of_camera,
            'max_column': self.max_column,
            'camera_infos': [camera_info.to_dict() for camera_info in self.log_infos]
            # Convert each Address to a dictionary
        }

    @classmethod
    def from_dict(cls, config_info):
        log_infos = [LogInfo.from_dict(log_info) for log_info in config_info['log_infos']]
        return cls(config_info['no_of_camera'], config_info['max_column'], log_infos)

def save_config_file(config_info_class, config_file_path):
    # Convert the Person instance to a dictionary
    config_info_data = config_info_class.to_dict()
    # Write the dictionary to a JSON file
    with open(config_file_path, 'w', encoding="utf-8") as file:
        json.dump(config_info_data, file, ensure_ascii=False, indent=4)


def load_config_log_file(file_path):
    # Read the JSON file and convert it back to a dictionary
    with open(file_path, 'r', encoding='utf-8') as file:
        config_info_data_read = json.load(file)
    # Create an instance of Person from the dictionary
    config_info_class = ConfigInfo.from_dict(config_info_data_read)
    return config_info_class

