import cv2
import streamlit as st
import dlib
import pandas as pd
from math import hypot

BLINK_THRESH = 6.0

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)

    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio


@st.cache
def load_model():

    landmark_d = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_detector = dlib.get_frontal_face_detector()
    return landmark_d,face_detector

ld,fd = load_model()


alert= st.empty()
st.title('Drowsiness Detection')
cam = cv2.VideoCapture(0)
start = st.checkbox('Start')
st.sidebar.text('Left Eye Ratio')
l_ratio = st.sidebar.empty()
st.sidebar.text('Right Eye Ratio')
r_ratio = st.sidebar.empty()
st.sidebar.text('Blink Ratio')
ratio = st.sidebar.empty()

img = st.empty()
chart=st.sidebar.empty()

df = pd.DataFrame(columns=['L ratio','R ratio' ,'Blink ratio'])

while start:

        _,frame = cam.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        faces = fd(gray)
        for face in faces:
            face_landmarks = ld(frame,face)
            
            left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], face_landmarks)
            right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], face_landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
            if blinking_ratio >= BLINK_THRESH:
                alert.warning('ALERT DONT SLEE')
            
         
            l_ratio.subheader(str(left_eye_ratio))
            r_ratio.subheader(str(right_eye_ratio))
            ratio.subheader(str(blinking_ratio))
            
            df=df.append([left_eye_ratio,right_eye_ratio,blinking_ratio],ignore_index=True)
            df=df.drop(axis=0,index=0)
            chart.line_chart(df)
            




        img.image(frame,"Cam")
img = st.empty()
cam.release()
