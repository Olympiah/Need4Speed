# Python In-built packages
from pathlib import Path
import PIL
from PIL import Image

# External packages
import streamlit as st
import pandas as pd

# Local Modules
import settings
import helper_app as helper

icon = Image.open('images/eye.png')
# Setting page layout
st.set_page_config(
    page_title="EyeSpeeD",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("EyeSpeeD")

# Sidebar
st.sidebar.header("Choose a model to load")

# Model Options
model_type = st.sidebar.radio(
    "Select model", ['License Plate Detection', 'Speed Detection'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or License Plate
# if model_type == 'Detection':
#     model_path = Path(settings.DETECTION_MODEL)
if model_type == 'License Plate Detection':
    model_path = Path(settings.LICENSE_PLATE_MODEL)
elif model_type == 'Speed Detection':
    model_path = Path(settings.SPEED_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# st.sidebar.header("Image/Video Config")
# source_radio = st.sidebar.radio(
#     "Select Source", settings.SOURCES_LIST)
#
# source_img = None
# # If image is selected
# if source_radio == settings.IMAGE:
#     source_img = st.sidebar.file_uploader(
#         "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
#
#     col1, col2 = st.columns(2)
#
#     with col1:
#         try:
#             if source_img is None:
#                 default_image_path = str(settings.DEFAULT_IMAGE)
#                 default_image = PIL.Image.open(default_image_path)
#                 st.image(default_image_path, caption="Default Image",
#                          use_column_width=True)
#             else:
#                 uploaded_image = PIL.Image.open(source_img)
#                 st.image(source_img, caption="Uploaded Image",
#                          use_column_width=True)
#         except Exception as ex:
#             st.error("Error occurred while opening the image.")
#             st.error(ex)
#
#     with col2:
#         if source_img is None:
#             default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
#             default_detected_image = PIL.Image.open(
#                 default_detected_image_path)
#             st.image(default_detected_image_path, caption='Detected Image',
#                      use_column_width=True)
#         else:
#             if st.sidebar.button('Detect Objects'):
#                 res = model.predict(uploaded_image,
#                                     conf=confidence
#                                     )
#                 boxes = res[0].boxes
#                 res_plotted = res[0].plot()[:, :, ::-1]
#                 st.image(res_plotted, caption='Detected Image',
#                          use_column_width=True)
#                 try:
#                     with st.expander("Detection Results"):
#                         for box in boxes:
#                             st.write(box.data)
#                 except Exception as ex:
#                     # st.write(ex)
#                     st.write("No image is uploaded yet!")
#
# elif source_radio == settings.VIDEO:
#     helper.play_stored_video(confidence, model)
#
# elif source_radio == settings.WEBCAM:
#     helper.play_webcam(confidence, model)
#
# elif source_radio == settings.RTSP:
#     helper.play_rtsp_stream(confidence, model)
#
# # elif source_radio == settings.YOUTUBE:
# #     helper.play_youtube_video(confidence, model)
#
# elif source_radio == settings.Video:
#     helper.license_detector(confidence, model)
#
#
#
#
# else:
#     st.error("Please select a valid source type!")

st.sidebar.header("TASK")
source_radio1 = st.sidebar.radio("Select task", settings.SOURCES_LIST2)
if source_radio1 == settings.LICENSE_PLATE_EXTRACTION:
    helper.license_detector(confidence, model)
    df = pd.read_csv("results1.csv")
    st.write(df)
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(df)
    st.download_button(
        "Download Data",
        csv,
        "results1.csv",
        key = 'download-csv'
    )
elif source_radio1 == settings.SPEED:
    helper.speed(confidence, model)
    f = pd.read_csv("results2.csv")
    st.write(f)
    def convert_f(f):
        return f.to_csv(index=False).encode('utf-8')
    csv2 = convert_f(f)
    st.download_button(
        "Download Data",
        csv2,
        "results2.csv",
        key = 'download-csv'
    )
else:
    st.error("Please select a valid task!")