from pathlib import Path
import sys
import os

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'
RTSP = 'RTSP'
YOUTUBE = 'YouTube'
LICENSE_PLATE_EXTRACTION = 'License Plate Extraction'
SPEED = 'Speed Detection'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, RTSP, YOUTUBE]
SOURCES_LIST2 = [LICENSE_PLATE_EXTRACTION, SPEED]
# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'car1.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'car1_detected.jpg'

# Videos config
# C:\Users\HP\PycharmProjects\Need4Speed\data
path = os.path.join('.', 'data', 'vid2.mp4')
# VIDEO_DIR = 'data'
VIDEO_1_PATH = path
# VIDEO_2_PATH = VIDEO_DIR / 'video_2.mp4'
# VIDEO_3_PATH = VIDEO_DIR / 'video_3.mp4'
# VIDEO_4_PATH = VIDEO_DIR / 'video_4.mp4'
# VIDEO_5_PATH = VIDEO_DIR / 'video_5.mp4'
# VIDEO_6_PATH = VIDEO_DIR / 'video_6.mp4'
# VIDEO_7_PATH = VIDEO_DIR / 'video_7.mp4'
# VIDEO_8_PATH = VIDEO_DIR / 'video_8.mp4'
# VIDEO_9_PATH = VIDEO_DIR / 'video_9.mp4'
# VIDEO_10_PATH = VIDEO_DIR / 'video_10.mp4'
# VIDEO_11_PATH = VIDEO_DIR / 'video_11.mp4'
# VIDEO_12_PATH = VIDEO_DIR / 'video_12.mp4'
# VIDEO_13_PATH = VIDEO_DIR / 'video_13.mp4'
# VIDEO_14_PATH = VIDEO_DIR / 'video_14.mp4'
VIDEOS_DICT = {
    'video_1': VIDEO_1_PATH,
    # 'video_2': VIDEO_2_PATH,
    # 'video_3': VIDEO_3_PATH,
    # 'video_4': VIDEO_4_PATH,
    # 'video_5': VIDEO_5_PATH,
    # 'video_6': VIDEO_6_PATH,
    # 'video_7': VIDEO_7_PATH,
    # 'video_8': VIDEO_8_PATH,
    # 'video_9': VIDEO_9_PATH,
    # 'video_10': VIDEO_10_PATH,
    # 'video_11': VIDEO_11_PATH,
    # 'video_12': VIDEO_12_PATH,
    # 'video_13': VIDEO_13_PATH,
    # 'video_14': VIDEO_14_PATH,
}

# ML Model config
# MODEL_DIR = path = os.path.join('.', 'data', 'vid2.mp4')
# DETECTION_MODEL = os.path.join('.', 'Models', 'vehicle_detect.pt')
LICENSE_PLATE_MODEL = os.path.join('.', 'Models', 'license.pt')
SPEED_MODEL = os.path.join('.', 'Models', 'yolov8n.pt')

# Webcam
WEBCAM_PATH = 0