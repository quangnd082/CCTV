import logging
import colorlog
import os
import glob
from datetime import datetime
from PyQt5.QtCore import QObject, pyqtSignal
import csv
import cv2
import numpy as np


class Logger(QObject):
    signalLog = pyqtSignal(str)

    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.name = name
        self.__logger = logging.getLogger(name)
        self.__logger.setLevel(logging.DEBUG)

        # Tạo thư mục lưu log nếu chưa có
        os.makedirs('Log_Vision/Log_View', exist_ok=True)
        os.makedirs('Log_Vision/Log_CSV', exist_ok=True)
        os.makedirs('Log_Vision/Log_Image', exist_ok=True)

        # Biến lưu ngày hiện tại để kiểm tra thay đổi file log
        self.current_date = datetime.now().strftime('%Y_%m_%d')

        # Tạo các handler cho log
        self.__create_log_handlers()

    def __create_log_handlers(self):
        """Tạo các handler cho log mỗi ngày."""
        today = datetime.now().strftime('%Y_%m_%d')
        log_path = f'Log_Vision/Log_View/{today}.log'

        # Formatter có màu cho console
        self.__formatter_color = colorlog.ColoredFormatter(
            fmt='%(log_color)s %(asctime)s %(name)s - %(filename)s - %(levelname)s - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red'  # Thay 'magenta' bằng 'bold_red'
            }
        )

        # Formatter không màu cho file log
        self.__formatter_no_color = logging.Formatter(
            fmt='%(asctime)s %(name)s - %(filename)s - %(levelname)s - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S'
        )

        # Xóa các handler cũ trước khi thêm mới
        self.__logger.handlers.clear()

        # File log dạng text
        log_handle = logging.FileHandler(log_path, encoding='utf-8')
        log_handle.setLevel(logging.DEBUG)
        log_handle.setFormatter(self.__formatter_no_color)
        self.__logger.addHandler(log_handle)

        # Hiển thị log ra console
        stream_handle = logging.StreamHandler()
        stream_handle.setLevel(logging.DEBUG)
        stream_handle.setFormatter(self.__formatter_color)
        self.__logger.addHandler(stream_handle)

    def __check_date_and_update_log(self):
        """Kiểm tra nếu ngày thay đổi thì tạo file log mới."""
        new_date = datetime.now().strftime('%Y_%m_%d')
        if new_date != self.current_date:
            self.current_date = new_date
            self.__create_log_handlers()

    def __emit_formatted_log(self, level, message):
        """Gửi tín hiệu log với format đầy đủ."""
        self.__check_date_and_update_log()

        # Tạo LogRecord
        record = logging.LogRecord(
            name=self.__logger.name,
            level=level,
            pathname=__file__,
            lineno=0,
            msg=message,
            args=None,
            exc_info=None
        )

        # Format log
        formatted_message = self.__formatter_no_color.format(record)

        # Gửi tín hiệu đã format
        self.signalLog.emit(formatted_message)

    def __manage_image_count(self, root_folder, max_images=500, delete_count=50):
        """Quản lý số lượng ảnh trong tất cả thư mục con"""
        try:
            # Tìm tất cả file ảnh trong thư mục gốc và các thư mục con
            image_files = glob.glob(os.path.join(root_folder, "**", "*.jpg"), recursive=True)
            total_images = len(image_files)
                        
            if total_images > max_images:
                # Sắp xếp theo thời gian tạo (cũ nhất trước)
                image_files.sort(key=os.path.getctime)
                
                # Xóa những ảnh cũ nhất
                deleted_count = 0
                for i in range(min(delete_count, total_images)):
                    try:
                        os.remove(image_files[i])
                        deleted_count += 1
                    except Exception as e:
                        self.warning(f"Cannot delete {image_files[i]}: {e}")
                
                self.info(f"Deleted {deleted_count} old images from {root_folder} (was {total_images}, now {total_images - deleted_count})")
            else:
                pass
                
        except Exception as e:
            self.error(f"Error managing image count: {e}")

    def log_image(self, model_name, image, image_folder, ret='PASS', log_csv=False):
        """Lưu ảnh vào thư mục log, ghi log thông tin ảnh và lưu log vào file CSV."""
        self.__check_date_and_update_log()
        date_str = datetime.now().strftime('%Y_%m_%d')
        time_str = datetime.now().strftime('%H_%M_%S')

        os.makedirs(image_folder, exist_ok=True)

        # Quản lý số lượng ảnh trước khi lưu
        self.__manage_image_count('Images/ImageCapture')
        self.__manage_image_count('Images/ImageTest')
        self.__manage_image_count('res/Database/Images/Source')
        self.__manage_image_count('Log_Vision/Log_Image')

        # Tạo tên file gốc
        base_filename = f'{ret}_{date_str}_{time_str}.jpg'
        image_path = os.path.join(image_folder, base_filename)
        
        # Kiểm tra và tạo tên file không trùng
        image_path = self._get_unique_filename(image_path)
        
        try:
            if image is None:
                raise ValueError("Invalid image (None). Cannot save")
            success = cv2.imwrite(image_path, image)

            if not success:
                raise IOError("Error saving image")

            image_path_fixed = image_path.replace("\\", "/")
            log_message = f'Image logged at {image_path_fixed}'
            self.info(log_message)

            if log_csv == True:
                # ✅ Ghi log vào file CSV
                csv_folder = 'Log_Vision/Log_CSV'
                os.makedirs(csv_folder, exist_ok=True)
                csv_path = os.path.join(csv_folder, f'{date_str}.csv')
                csv_exists = os.path.isfile(csv_path)

                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if not csv_exists:
                        writer.writerow(['Model', 'Result', 'Time'])
                    writer.writerow([model_name, ret, time_str])

            return image_path_fixed
        except Exception as ex:
            self.error(ex)

    def _get_unique_filename(self, filepath):
        """Tạo tên file duy nhất, thêm (1), (2), ... nếu file đã tồn tại"""
        if not os.path.exists(filepath):
            return filepath
        
        # Tách đường dẫn, tên file và extension
        directory = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)
        
        counter = 1
        while True:
            new_filename = f"{name}({counter}){ext}"
            new_filepath = os.path.join(directory, new_filename)
            
            if not os.path.exists(new_filepath):
                return new_filepath
            
            counter += 1

    # Các hàm log với đầy đủ format
    def debug(self, message):
        self.__emit_formatted_log(logging.DEBUG, str(message))
        self.__logger.debug(message, stacklevel=2)

    def info(self, message):
        self.__emit_formatted_log(logging.INFO, str(message))
        self.__logger.info(message, stacklevel=2)

    def warning(self, message, exc=None):
        self.__emit_formatted_log(logging.WARNING, message)
        self.__logger.warning(message, stacklevel=2, exc_info=exc)

    def error(self, message, exc=None):
        self.__emit_formatted_log(logging.ERROR, message)
        self.__logger.error(message, stacklevel=2, exc_info=exc)

    def critical(self, message, exc=None):
        self.__emit_formatted_log(logging.CRITICAL, message)
        self.__logger.critical(message, stacklevel=2, exc_info=exc)


if __name__ == '__main__':
    logger_1 = Logger('Quang')

    try:
        logger_1.debug('Test debug message')
        logger_1.info('Test info message')
        logger_1.warning('Test warning message')
        logger_1.error('Test error message')

        # Giả lập lưu ảnh
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        logger_1.log_image(model_name="YOLOv9", image=image)

        # Gây lỗi để test critical
        result = 1 / 0

    except Exception as e:
        logger_1.critical(f'Exception occurred: {e}')