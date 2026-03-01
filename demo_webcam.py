"""
EmotiScan — Live Webcam Emotion Demo
=====================================
Uses a pre-trained FER model for real-time emotion classification.

Usage:
    python demo_webcam.py

Controls:
    Q / Esc  →  Quit
    S        →  Save screenshot
"""

import cv2
import time
import threading

# ── Configuration ─────────────────────────────────────────────────────────────
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

EMOTION_COLORS = {
    'angry':    (0,   0,   220),
    'disgust':  (128, 0,   128),
    'fear':     (200, 80,  0  ),
    'happy':    (0,   210, 210),
    'sad':      (200, 150, 0  ),
    'surprise': (0,   165, 255),
    'neutral':  (150, 150, 150),
}


def draw_emotion_bars(frame, x, y, w, emotions):
    bar_x     = x + w + 12
    bar_h     = 20
    bar_max_w = 130
    padding   = 4

    panel_h = (bar_h + padding) * 7 + padding
    overlay = frame.copy()
    cv2.rectangle(overlay, (bar_x - 4, y - 2), (bar_x + bar_max_w + 4, y + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    for i, emotion in enumerate(EMOTION_LABELS):
        prob  = emotions.get(emotion, 0.0)
        top   = y + i * (bar_h + padding)
        color = EMOTION_COLORS[emotion]

        filled = int(prob * bar_max_w)
        cv2.rectangle(frame, (bar_x, top), (bar_x + bar_max_w, top + bar_h), (60, 60, 60), -1)
        if filled > 0:
            cv2.rectangle(frame, (bar_x, top), (bar_x + filled, top + bar_h), color, -1)

        label = f"{emotion.capitalize()[:3]}  {prob*100:.0f}%"
        cv2.putText(frame, label, (bar_x + 4, top + bar_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)


def draw_loading_screen(frame, dots):
    """Overlay a 'Loading model...' message on the live feed."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    text  = "Loading EmotiScan" + "." * dots
    scale = 1.0
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    cx, cy = w // 2 - tw // 2, h // 2
    cv2.putText(frame, text, (cx, cy),
                cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    cv2.putText(frame, "EmotiScan", (cx + tw // 2 - 80, cy - 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 210, 210), 2, cv2.LINE_AA)
    cv2.putText(frame, "Real-Time Emotion Recognition", (w // 2 - 160, cy + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)


def run_demo():
    # ── Open webcam immediately ───────────────────────────────────────────────
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Make sure it's not in use by another app.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── Load FER model in background thread ───────────────────────────────────
    detector: list = [None]
    model_ready = threading.Event()

    def load_model():
        from fer.fer import FER  # type: ignore
        detector[0] = FER(mtcnn=False)
        model_ready.set()

    threading.Thread(target=load_model, daemon=True).start()

    print("=== EmotiScan Live Demo ===")
    print("Press Q or Esc to quit, S to save a screenshot.\n")

    fps_counter = 0
    fps_start   = time.time()
    fps_display = 0.0
    dot_counter = 0
    dot_timer   = time.time()
    results     = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        if not model_ready.is_set():
            # Animate dots while loading
            if time.time() - dot_timer > 0.4:
                dot_counter = (dot_counter + 1) % 4
                dot_timer = time.time()
            draw_loading_screen(frame, dot_counter)

        else:
            # ── Run emotion detection ─────────────────────────────────────────
            results = detector[0].detect_emotions(frame)

            for face in results:
                x, y, w, h = face['box']
                emotions   = face['emotions']
                dominant   = max(emotions, key=emotions.get)
                confidence = emotions[dominant]
                color      = EMOTION_COLORS[dominant]

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

                label_text = f"{dominant.capitalize()}  {confidence*100:.0f}%"
                (tw, _th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
                cv2.rectangle(frame, (x, y - _th - 14), (x + tw + 12, y), color, -1)
                cv2.putText(frame, label_text, (x + 6, y - 8),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

                draw_emotion_bars(frame, x, y, w, emotions)

            # ── Header bar ───────────────────────────────────────────────────
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_start   = time.time()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (370, 36), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame,
                        f"EmotiScan  |  FPS: {fps_display}  |  Faces: {len(results)}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('EmotiScan — Real-Time Emotion Recognition', frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
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
