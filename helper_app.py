from ultralytics import YOLO
# import time, datetime, string
import streamlit as st
import cv2, cvzone
from deep_sort_realtime.deepsort_tracker import DeepSort
from sort.sort import *
# from pytube import YouTube
from helper import write_csv, get_vehicle, read_license_plate, csv, estimatedSpeed, get_class_color
import numpy as np
import settings
import math

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


# def play_youtube_video(conf, model):
#     """
#     Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.
#
#     Parameters:
#         conf: Confidence of YOLOv8 model.
#         model: An instance of the `YOLOv8` class containing the YOLOv8 model.
#
#     Returns:
#         None
#
#     Raises:
#         None
#     """
#     source_youtube = st.sidebar.text_input("YouTube Video url")
#
#     is_display_tracker, tracker = display_tracker_options()
#
#     if st.sidebar.button('Detect Objects'):
#         try:
#             yt = YouTube(source_youtube)
#             stream = yt.streams.filter(file_extension="mp4", res=720).first()
#             vid_cap = cv2.VideoCapture(stream.url)
#
#             st_frame = st.empty()
#             while (vid_cap.isOpened()):
#                 success, image = vid_cap.read()
#                 if success:
#                     _display_detected_frames(conf,
#                                              model,
#                                              st_frame,
#                                              image,
#                                              is_display_tracker,
#                                              tracker,
#                                              )
#                 else:
#                     vid_cap.release()
#                     break
#         except Exception as e:
#             st.sidebar.error("Error loading video: " + str(e))


def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    # vid_cap = cv2.VideoCapture(source_rtsp)
                    # time.sleep(0.1)
                    # continue
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys(), key='vid')

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Run App'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )

                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def license_detector(conf, model):
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys(), key='vid2')

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Run App'):
        try:
            GREEN = (0, 255, 0)
            WHITE = (255, 255, 255)
            vehicle_detector = YOLO('Models/vehicle_detect.pt')
            license_plate_detector = model
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            results = {}
            mot_tracker = Sort()
            st_frame = st.empty()
            frame_nmr = -1
            ret = True
            while ret and frame_nmr < 20:
                frame_nmr += 1
                ret, frame = vid_cap.read()
                if ret:
                    results[frame_nmr] = {}
                    detections = vehicle_detector(frame)[0]
                    detections_ = []
                    for detection in detections.boxes.data.tolist():

                        x1, y1, x2, y2, score, class_id = detection
                        confidence = score
                        if float(confidence) < conf:
                            continue
                        x1 = int(x1)
                        y1 = int(y1)
                        x2 = int(x2)
                        y2 = int(y2)
                        detections_.append([x1, y1, x2, y2, score])
                    tracks = mot_tracker.update(np.asarray(detections_))
                    license_plates = license_plate_detector(frame)[0]
                    for license_plate in license_plates.boxes.data.tolist():
                        # unwrapping the detections
                        x1, y1, x2, y2, score, class_id = license_plate
                        x1 = int(x1)
                        y1 = int(y1)
                        x2 = int(x2)
                        y2 = int(y2)
                        cv2.line(frame, (x1, y1), (x1 + 200, y1), (0.255, 0), 10)
                        cv2.line(frame, (x1, y2), (x1 + 200, y2), (0.255, 0), 10)
                        cv2.line(frame, (x2, y1), (x2, y1 + 200), (0.255, 0), 10)
                        cv2.line(frame, (x2, y2), (x2 - 200, y2), (0.255, 0), 10)
                        if float(score) < conf:
                            continue
                        xcar1, ycar1, xcar2, ycar2, vehicle_id = get_vehicle(license_plate, tracks)
                        if vehicle_id != -1:
                            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255,
                                                                         cv2.THRESH_BINARY_INV)
                            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                            if license_plate_text is not None:
                                results[frame_nmr][vehicle_id] = {'vehicle': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                                    'text': license_plate_text,
                                                                                    'bbox_score': score,
                                                                                    'text_score': license_plate_text_score}}

                write_csv(results, './results1.csv')
                _display_detected_frames(conf, model, st_frame, frame, is_display_tracker, tracker)

            else:
                vid_cap.release()

            # print(results)
            # write results
            # write_csv(results, './results1.csv')

        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def speed(conf, model):
    source_vid3 = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())
    is_display_tracker, tracker = display_tracker_options()
    with open(settings.VIDEOS_DICT.get(source_vid3), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Estimate Speed'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid3)))
            smodel = model
            # mask = cv2.imread("images/mask.png")
            tracker1 = DeepSort(
                max_iou_distance=0.7,
                max_age=2,
                n_init=3,
                nms_max_overlap=3.0,
                max_cosine_distance=0.2)
            coordinatesDict = dict()  # Define a dictionary which will hold the coordinates/location of the detected objects
            st_frame = st.empty()
            frame_nmr = 0
            speedres = {}
            ret = True
            while True:
                success, img = vid_cap.read()
                frame_nmr += 1
                speedres[frame_nmr] = {}
                img = cv2.resize(img, (1280, 720))
                # imgRegion = cv2.bitwise_and(img, mask)
                results = smodel(img, stream=True)
                detections = list()
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1
                        bbox = (x1, y1, w, h)
                        # Confidence
                        conf = math.ceil((box.conf[0] * 100)) / 100

                        cls = int(box.cls[0])

                        currentClass = smodel.names[cls]
                        if currentClass == 'car' and conf > 0.5:
                            w, h = x2 - x1, y2 - y1
                            detections.append(([x1, y1, w, h], conf, cls))

                        elif currentClass == "truck":
                            w, h = x2 - x1, y2 - y1
                            detections.append(([x1, y1, w, h], conf, cls))

                        elif currentClass == "motorbike":
                            w, h = x2 - x1, y2 - y1
                            detections.append(([x1, y1, w, h], conf, cls))

                tracks = tracker1.update_tracks(detections, frame=img)  # Initialize tracks
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id

                    bbox = track.to_ltrb()  # Convert to left, top, right, bottom format for bbox
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    w, h = x2 - x1, y2 - y1

                    co_ord = [x1, y1]

                    if track_id not in coordinatesDict:
                        coordinatesDict[track_id] = co_ord
                    else:
                        if len(coordinatesDict[track_id]) > 2:
                            del coordinatesDict[track_id][-3:-1]
                        coordinatesDict[track_id].append(co_ord[0])
                        coordinatesDict[track_id].append(co_ord[1])
                    estimatedSpeedValue = 0
                    if len(coordinatesDict[track_id]) > 2:
                        location1 = [coordinatesDict[track_id][0], coordinatesDict[track_id][2]]
                        location2 = [coordinatesDict[track_id][1], coordinatesDict[track_id][3]]
                        estimatedSpeedValue = estimatedSpeed(location1, location2)
                        if estimatedSpeedValue is not None:
                            speedres[frame_nmr][track_id] = {'vehicle': {'speed': estimatedSpeedValue}}

                    cls = track.get_det_class()
                    currentClass = smodel.names[cls]
                    clsColor = get_class_color(currentClass)

                    cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2)

                    cvzone.putTextRect(
                        img,
                        text=f"{smodel.names[cls]} {estimatedSpeedValue} km/h",
                        pos=(max(0, x1), max(35, y1)),
                        scale=1,
                        thickness=1,
                        offset=2)

                    cx, cy = x1 + w // 2, y1 + h // 2

                    cv2.circle(img, (cx, cy), radius=5, color=clsColor, thickness=cv2.FILLED)
                csv(speedres, './results2.csv')
                res = smodel(img, conf=conf)
                res_plotted = res[0].plot()
                st_frame.image(res_plotted,
                               caption='Detected Video',
                               channels="BGR",
                               use_column_width=True
                               )
                # cv2.imshow('Speed Detection', img)
                # cv2.waitKey(1)
                # if cv2.waitKey(1)&0xFF==27:
                #     break
            # else:
# vid_cap.release()
# cv2.destroyAllWindows()

        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
