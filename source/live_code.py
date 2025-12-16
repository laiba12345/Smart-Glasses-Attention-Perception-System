import cv2
import numpy as np
import sounddevice as sd
import threading
import queue
from collections import deque
from face import face_recognition
from faster_whisper import WhisperModel
from fer import FER
from ultralytics import YOLO

# ================= CONFIG =================
MODEL_SIZE = "tiny"
DEVICE = "cpu"
SAMPLE_RATE = 16000
CHUNK_DURATION = 3.0  # seconds
CAMERA_INDEX = 0

# ================= LOAD MODELS =================
print("[INFO] Loading models...")

whisper_model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type="int8"
)

yolo_model = YOLO("models/yolov8n-face.pt")


emotion_detector = FER(mtcnn=True)


# Emotion colors
emotion_colors = {
    'angry': (0, 0, 255),
    'disgust': (0, 128, 0),
    'fear': (128, 0, 128),
    'happy': (0, 255, 255),
    'sad': (255, 0, 0),
    'surprise': (255, 165, 0),
    'neutral': (200, 200, 200)
}

def draw_multiline_text(frame, text, x, y, font, font_scale, color, thickness, max_width):
    """Draw text on frame and wrap lines that exceed max_width."""
    words = text.split()
    line = ""
    y_offset = 0

    for word in words:
        # Test line width
        test_line = f"{line} {word}".strip()
        (w, h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if w > max_width:
            # Draw current line
            cv2.putText(frame, line, (x, y + y_offset), font, font_scale, color, thickness)
            line = word
            y_offset += h + 5  # move to next line
        else:
            line = test_line

    # Draw the last line
    if line:
        cv2.putText(frame, line, (x, y + y_offset), font, font_scale, color, thickness)


print("[INFO] Models loaded.")

# ================= QUEUES =================
audio_queue = queue.Queue()
text_queue = queue.Queue()

# ================= AUDIO RECORDING THREAD =================
def record_audio():
    while True:
        audio = sd.rec(
            int(CHUNK_DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )
        sd.wait()
        audio_queue.put(audio[:, 0].copy())

threading.Thread(target=record_audio, daemon=True).start()

# ================= TRANSCRIPTION THREAD =================
def transcribe_loop():
    buffer = deque(maxlen=int(SAMPLE_RATE * 3))  # rolling 3s buffer

    while True:
        chunk = audio_queue.get()
        buffer.extend(chunk)

        audio_np = np.array(buffer, dtype=np.float32)

        segments, _ = whisper_model.transcribe(
            audio_np,
            beam_size=1,
            language="en"
        )

        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        if text:
            text_queue.put(text)

threading.Thread(target=transcribe_loop, daemon=True).start()

# ================= WEBCAM LOOP =================
cap = cv2.VideoCapture(CAMERA_INDEX)
latest_text = ""

print("[INFO] Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------- Update transcription --------
    while not text_queue.empty():
        latest_text = text_queue.get()

    # -------- Face detection (YOLO) --------
    results = yolo_model(frame, verbose=False)[0]

    faces = []
    if results.boxes is not None:
        faces = results.boxes.xyxy.cpu().numpy()

    # -------- Emotion detection --------

        dominant_emotions = []
        for (xmin, ymin, xmax, ymax) in faces:
            xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
            h, w = frame.shape[:2]
    
    # Add padding
            dx = int(0.1 * (xmax - xmin))
            dy = int(0.1 * (ymax - ymin))
            xmin_pad = max(0, xmin - dx)
            ymin_pad = max(0, ymin - dy)
            xmax_pad = min(w, xmax + dx)
            ymax_pad = min(h, ymax + dy)
    
            face_img = frame[ymin_pad:ymax_pad, xmin_pad:xmax_pad]
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
            dominant_emotion = "Neutral"
            try:
                print("Face crop shape:", face_img_rgb.shape)
                emotions = emotion_detector.detect_emotions(face_img_rgb)
                print("Detected Emotions:", emotions)
                if emotions:
                    dominant_emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
            except Exception as e:
                print("FER error:", e)
    
            dominant_emotions.append(dominant_emotion)

     #---------Face Recognition -----------
        face_recognitions = []
        for (xmin, ymin, xmax, ymax) in faces:
            xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
            face_img = frame[ymin:ymax, xmin:xmax]
            detected = face_recognition(face_img)
            face_recognitions.append(detected["pred"])

    # -------- Draw faces + emotions --------
    for (bbox, dom_em, identity) in zip(faces, dominant_emotions, face_recognitions):
       xmin, ymin, xmax, ymax = bbox
       xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])  # ensure integers

       # Ensure dom_em is string
       dom_em_str = str(dom_em).upper() if dom_em is not None else "NEUTRAL"

       color = emotion_colors.get(dom_em_str.lower(), (255, 255, 255))
    
       # Draw emotion
       cv2.putText(frame, dom_em_str, (xmin, max(15, ymin-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
       # Draw recognized identity (above the emotion)
       identity_str = str(identity) if identity is not None else "Unknown"
       cv2.putText(frame, identity_str, (xmin, max(15, ymin-25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    # -------- Overlay transcription --------
    y0 = 30 
    max_width = frame.shape[1] - 20  # leave 10px margin
    draw_multiline_text(frame, latest_text, 10, 30, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, max_width)

        
    cv2.imshow("Live Transcription + Emotion Detection + Face Recognition", frame)
    
    # ===== WaitKey is required to refresh the window =====
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()
print("[INFO] Shutdown complete.")
