import cv2
import tensorflow as tf 
import streamlit as st 
import dlib
import pandas as pd
import numpy as np

label = ["Yawn", "No_Yawn", "Closed", "Open"]
df = pd.DataFrame(columns=label)
IMG_SIZE = 145
def prepare(img):
    
    img_array = img / 255
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


@st.cache
def load_model():

    model = tf.keras.models.load_model('drowiness_vgg16.h5')
    return model
model = load_model()

alert= st.empty()
st.title('Drowsiness Detection')
cam = cv2.VideoCapture(0)
start = st.checkbox('Start')
textside = st.sidebar.empty()

img = st.empty()
chart=st.sidebar.empty()
while start:

        _,frame = cam.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        prediction = model.predict([prepare(frame)])
       
        p= label[np.argmax(prediction)]
        df=df.append({
            "Yawn":prediction[0][0],
            "No_Yawn":prediction[0][1],
            "Closed":prediction[0][2],
            "Open":prediction[0][3],
        },ignore_index=True)
        
        #df=df.drop(axis=0,index=0)
        chart.line_chart(df)
       
        textside.subheader("Label :"+str(p))
        if p=='Yawn' or 'Closed':
            alert.warning('ALERT DONT SLEEP')
        
        
        
        img.image(frame,"Cam")
img = st.empty()
cam.release()