import os
import av
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Get the absolute path to the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "emotion_model.h5")

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file 'emotion_model.h5' not found. Please upload it.")
else:
    model = load_model(MODEL_PATH)

# Load AI Model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def detect_emotion(frame):
    """Detects emotion from a given frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    emotion_label = "Neutral"
    for (x, y, w, h) in faces:
        roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48)) / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        emotion_label = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"][np.argmax(model.predict(roi))]
    return emotion_label

def analyze_attention(frame):
    """Tracks attention by detecting gaze movement."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    return "Focused" if results.multi_face_landmarks else "Distracted"

class VideoTransformer(VideoTransformerBase):
    """Processes the webcam video stream."""
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        emotion = detect_emotion(img)
        attention = analyze_attention(img)

        cv2.putText(img, f"Emotion: {emotion}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, f"Attention: {attention}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return img

def main():
    """Runs the Streamlit UI."""
    st.title("🎭 AI-Powered Emotion & Attention Tracking for ADHD")
    st.write("Real-time emotion detection and attention tracking simulation")

    # Use Streamlit WebRTC for video streaming
    webrtc_streamer(key="camera", video_transformer_factory=VideoTransformer)

if __name__ == "__main__":
    main()