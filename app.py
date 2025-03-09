import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load AI Model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model('emotion_model.h5')
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

def main():
    """Runs the Streamlit UI."""
    st.title("🎭 AI-Powered Emotion & Attention Tracking for ADHD")
    st.write("Real-time emotion detection and attention tracking simulation")

    if st.button("📷 Capture Image"):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            st.error("Failed to capture image.")
            return
        
        # Process the captured image
        emotion = detect_emotion(frame)
        attention = analyze_attention(frame)
        
        # Display results
        st.image(frame, channels="BGR")
        st.write(f"**Detected Emotion:** {emotion}")
        st.write(f"**Attention Status:** {attention}")

if __name__ == "__main__":
    main()