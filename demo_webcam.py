"""
EmotiScan — Live Webcam Emotion Demo
=====================================
Run this script after training to classify emotions from your webcam in real-time.

Usage:
    python demo_webcam.py

Controls:
    Q  →  Quit
    S  →  Save screenshot
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import os

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH = 'saved_model/emotiscan_cnn.h5'
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Colors in BGR
EMOTION_COLORS = {
    'Angry':    (0,   0,   220),   # Red
    'Disgust':  (128, 0,   128),   # Purple
    'Fear':     (200, 100, 0  ),   # Dark blue-ish
    'Happy':    (0,   200, 200),   # Yellow
    'Sad':      (200, 150, 0  ),   # Teal-ish
    'Surprise': (0,   165, 255),   # Orange
    'Neutral':  (150, 150, 150),   # Gray
}

EMOJI = {
    'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨',
    'Happy': '😄', 'Sad': '😢', 'Surprise': '😲', 'Neutral': '😐'
}


def preprocess_face(face_img):
    """Resize and normalize a face crop for the model."""
    face = cv2.resize(face_img, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=(0, -1))  # (1, 48, 48, 1)
    return face


def draw_emotion_bar(frame, x, y, w, probs, label):
    """Draw a confidence bar chart next to the detected face."""
    bar_x = x + w + 10
    bar_y = y
    bar_height = 18
    bar_max_width = 120

    for i, (emotion, prob) in enumerate(zip(EMOTION_LABELS, probs)):
        top = bar_y + i * (bar_height + 3)
        color = EMOTION_COLORS[emotion]

        # Background
        cv2.rectangle(frame,
                      (bar_x, top),
                      (bar_x + bar_max_width, top + bar_height),
                      (50, 50, 50), -1)

        # Filled bar
        filled = int(prob * bar_max_width)
        cv2.rectangle(frame,
                      (bar_x, top),
                      (bar_x + filled, top + bar_height),
                      color, -1)

        # Label + percentage
        text = f'{emotion[:3]} {prob*100:.0f}%'
        cv2.putText(frame, text,
                    (bar_x + 3, top + bar_height - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (255, 255, 255), 1, cv2.LINE_AA)


def run_demo():
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at '{MODEL_PATH}'")
        print("Please run EmotiScan_Notebook.ipynb first to train and save the model.")
        return

    print("Loading EmotiScan model...")
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded!")

    # Load face detector
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("\n=== EmotiScan Live Demo ===")
    print("Press Q to quit, S to save a screenshot.\n")

    fps_counter = 0
    fps_start   = time.time()
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )

        for (fx, fy, fw, fh) in faces:
            # Crop face from original frame
            face_crop = frame[fy:fy+fh, fx:fx+fw]
            preprocessed = preprocess_face(face_crop)

            # Predict
            probs = model.predict(preprocessed, verbose=0)[0]
            pred_idx = np.argmax(probs)
            label    = EMOTION_LABELS[pred_idx]
            conf     = probs[pred_idx]
            color    = EMOTION_COLORS[label]

            # Draw bounding box
            thickness = 3
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), color, thickness)

            # Draw label pill
            label_text = f'{label}  {conf*100:.0f}%'
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
            cv2.rectangle(frame,
                          (fx, fy - th - 14),
                          (fx + tw + 12, fy),
                          color, -1)
            cv2.putText(frame, label_text,
                        (fx + 6, fy - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # Draw confidence bars
            draw_emotion_bar(frame, fx, fy, fw, probs, label)

        # FPS counter
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_start   = time.time()

        # Header overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (320, 34), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f'EmotiScan  |  FPS: {fps_display}  |  Faces: {len(faces)}',
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('EmotiScan — Real-Time Emotion Recognition', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            fname = f'screenshot_{int(time.time())}.png'
            cv2.imwrite(fname, frame)
            print(f"Screenshot saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("Demo ended.")


if __name__ == '__main__':
    run_demo()
