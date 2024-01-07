![image](https://github.com/Olympiah/Need4Speed/assets/87874970/6652d239-61de-428b-a9b6-57dbfe80d15d)# EYESPEED
This is a real-time vehicle speed estimation system built using Object Detection and Object Tracking 

<img src="screenshot.png" alt="Screenshot of the application" width="500"/>
<img src="screenshot.png" alt="Screenshot of the application" width="500"/>

## Overview
This project was divided into 4 main parts i.e.
- Vehicle Detection
- Vehicle Tracking
- License Plate Detection and Recognition
- Speed Estimation

Vehicle Detection

  This stage involved the use of Yolov8 trained on custom data. The output was detection details of the vehicles in the form of a list i.e. bounding boxes, confidence, class ids.
  
Vehicle Tracking

After detections are updated, the vehicles are then assigned a unique track id. The tracking object is appended to the detection list which now holds all vehicle relevant information.

License Plate Detection and Recognition

Here the license plates are detected, preprocessed and the output fed into the ocr model for text extraction. The output csv file then contains all the details necessary for vehicle identification

Speed Estimation 

This was calculated using the formula: Speed (in km/hr) = Distance Travelled by the detected trucks (in meters) * fps * 4
The distance of the the vehicle between frames was calculated using the euclidian distance formula, we found that the rate of pixels per metre that assured the highest auracy was 4.
The tracking object from the tracker was also useful in speed estimation. The output is a real-time video and csv file containing the speed details.

*Limitations*
1. The ocr model did not do well when the video input was of low quality thus not easily visible. This caused the system to crash
2. The scarcity of supervised Traffic Vehicle datasets in Kenya

## Requirements
* Python3
* OpenCV
* Deepsort Realtime
* Streamlit

## Table of Contents
* Algorithm
* Machine Learning
* Data 
* Screenshots of Application

  ## MACHINE LEARNING
  Below are the links to the notebooks in colab:
  * Vehicle Detection: https://colab.research.google.com/drive/1XWZogF_KHV0p3gGBUG2VXcdxxgp_VD2i?usp=sharing
  * License Plate Detection: https://colab.research.google.com/drive/1QTlWlczC9XDjGyLCQB6ERIRU0aTATkA_?usp=sharing

  ## Data
  I used various data sources. Below are the links to the various sources;
  
   License Plate Recognition - https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4
  
   Vehicle Detection -
  
   Speed Testing Dataset - https://slobodan.ucg.ac.me/science/vs13
  
   Traffic videos in Kenyan roads - https://drive.google.com/drive/folders/1-7KyU6zxIZAVlSHIgbPYHl3L3GOn1Lh_?usp=drive_link

  ## Screenshot
