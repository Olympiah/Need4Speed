from ultralytics import YOLO
import cv2
import os
import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort
from helper import get_vehicle, read_license_plate, write_csv

# define some constants
CONFIDENCE_THRESHOLD = 0.6
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# Dictionary where all the necessary info will be saved
results = {}

# mot_tracker = Sort()
tracker = DeepSort(max_age=50)

# load models i.e. both the vehicle detection model and license plate detection model
detect_path = os.path.join('.', 'Models', 'vehicle_detect.pt')
vehicle_detector = YOLO(detect_path)

model_path = os.path.join('.', 'Models', 'license.pt')
license_plate_detector = YOLO(model_path)

# importing path to the video
path = os.path.join('.', 'data', "vid1.mp4")

# load video i.e. initializing the video capture object
cap = cv2.VideoCapture(path)

# vehicle_class_idx = [0, 1, 2, 3, 4, 5, 6]

# read the frames
frame_nmr = -1
ret = True
# remember to remove the condition of 10 as it's just for testing
while ret and frame_nmr < 10:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}

        # detect vehicles
        detections = vehicle_detector(frame)[0]
        # print(detections)

        # Initializing an empty screen for detections
        detections_ = []

        ######################################
        # DETECTION
        ######################################
        for detection in detections.boxes.data.tolist():
            # output: bounding box coordinates, confidence score, class_id
            x1, y1, x2, y2, score, class_id = detection
            # if int(class_id) in vehicles:

            # extract the confidence (i.e., probability) associated with the prediction
            confidence = score

            # filter out weak detections by ensuring the
            # confidence is greater than the minimum confidence
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue
            # if the confidence is greater than the minimum confidence,
            # get the bounding box and the confidence score

            # Read them as integers
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # appending the bbx and confidence score
            detections_.append([[x1, y1, x2, y2], score])

        ######################################
        # TRACKING VEHICLES
        ######################################

        # NOTE: It contains the bbox of all vehicles updated bbox + tracking info i.e. track_id

        # update the tracker with the new detections
        # Perform Kalman filter measurement update step and update the feature cache.
        tracks = tracker.update_tracks(detections_, frame=frame)
        # print(tracks)

        # loop over the tracks
        for track in tracks:
            # if the track is not confirmed, ignore it
            # Returns True if this track is confirmed
            if not track.is_confirmed():
                continue

            # get the track id and the bounding box
            # track_id = track.track_id

            # Gets current position in bounding box format `(min x, miny, max x, max y)
            # returns the bounding box
            # ltrb = track.to_ltrb()

            # x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            # bbox = [[x1, x2, y1, y2]]

        ######################################
        # LICENSE PLATE DETECTION
        ######################################
        license_plates = license_plate_detector(frame)[0]
        # Looping over the license plate detections

        for license_plate in license_plates.boxes.data.tolist():
            # unwrapping the detections
            x1, y1, x2, y2, score, class_id = license_plate

            # extract the confidence (i.e., probability) associated with the prediction
            confidence = score

            # filter out weak detections by ensuring the
            # confidence is greater than the minimum confidence
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            # Assign a license plate to each detected vehicle
            # NOTE: At this point we have detected all vehicles in a frame and license plates as well

            # Return is vehicle coordinates/bbox and track_id
            xcar1, ycar1, xcar2, ycar2, vehicle_id = get_vehicle(license_plate, tracks)

            # If we have associated a vehicle id to a license plate
            if vehicle_id != -1:

                # crop license plate- wth license plate coordinates
                # getting the license plate alone yani
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                ######################################
                # PROCESSING LICENSE PLATE
                ######################################

                # Converting the license plate to a grayscale image
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                # If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to a maximum
                # value. The output is what we feed to the OCR

                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Visualizing both images
                # cv2.imshow('Cropped license plate', license_plate_crop)
                # cv2.imshow('Threshold plate', license_plate_crop_thresh)
                #
                # cv2.waitKey(0)

                # Reading the license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                # I f there is no trouble/error while reading LP then
                if license_plate_text is not None:
                    results[frame_nmr][vehicle_id] = {'vehicle': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                      'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}
        ######################################
        # SPEED ESTIMATION- yet to be implemented
        ######################################


# write results
write_csv(results, './results.csv')
