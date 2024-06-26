import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from keras_preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from subprocess import call

# load model
emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}
# load json and create model
json_file = open('emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
  
# load weights into new model
classifier.load_weights("emotion_model1.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(0, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        return img

def main():
    # Face Analysis Application #
    st.title("Real Time Face Sentiment and Drowsiness Detection Application")
    activiteis = ["Home", "Webcam Face Sentiment Detection","Face Drowsiness Detection","About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.markdown(
        """
    <style>
    .reportview-container .markdown-text-container {
        font-family: monospace;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    .Widget>label {
        color: white;
        font-family: monospace;
    }
    [class^="st-b"]  {
        color: white;
        font-family: monospace;
    }
    .st-bb {
        background-color: transparent;
    }
    .st-at {
        background-color: #0c0080;
    }
    footer {
        font-family: monospace;
    }
    .reportview-container .main footer, .reportview-container .main footer a {
        color: #0c0080;
    }
    header .decoration {
        background-image: none;
    }

    </style>
    """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """ Developed by Bless Dmello 9084  \n Niel Dmello 9086  \n Uttam Savaliya 8905
        """)
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Sentiment and Drowsiness Detection Application using OpenCV, Custom CNN model, Computer Vision System and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has two functionalities.

                 1. Real time Face Sentiment Detection using web cam feed.

                 2. Real time face Drowsiness Detection using web cam feed.

                 """)
    elif choice == "Webcam Face Sentiment Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your sentiment")
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Real time Face Sentiment and Drowsiness Detection Application using OpenCV, Custom Trained CNN model, Computer Vision System and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                             		<div style="background-color:#98AFC7;padding:10px">
                             		<h4 style="color:white;text-align:center;">This Application is developed by Bless Dmello(9084),Niel Dmello(9086) and Uttam Savaliya(8905) using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose. </h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    elif choice == "Face Drowsiness Detection":
        def open_eye():
            call(["python","Drowsiness_Detection.py"])
        open_eye()
    else:
        pass


if __name__ == "__main__":
    main()
