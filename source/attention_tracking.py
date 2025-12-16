# ---------------- Imports ----------------
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
from fer import FER
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel
from face import face_recognition
from tqdm import tqdm
import subprocess

# ---------------- Config ----------------
video_path = r"D:\Masters\ComputerVision\Project\data\video4.mp4"
processed_video_path = r"D:\Masters\ComputerVision\Project\data\result_no_audio.mp4"
final_output_path = r"D:\Masters\ComputerVision\Project\data\result_with_audio.mp4"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

# ---------------- Load Models ----------------
gazelle_model, transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitb14_inout')
gazelle_model.eval().to(device)

yolo = YOLO(r"models\best.pt")
face_model = YOLO("models\yolov8n-face.pt")
emotion_detector = FER(mtcnn=True)
whisper_model = WhisperModel("small", device='cpu', compute_type="int8")  # keep CPU for int8



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

inout_thresh = 0.5

# ---------------- Functions ----------------
def point_in_box(px, py, box):
    xmin, ymin, xmax, ymax = box
    return xmin <= px <= xmax and ymin <= py <= ymax


def visualize_gaze(pil_image, heatmaps, norm_bboxes, inout_scores, inout_thresh=0.5):
    colors = ['lime', 'tomato', 'cyan', 'fuchsia', 'yellow']
    overlay_image = pil_image.convert("RGBA")
    draw = ImageDraw.Draw(overlay_image)
    width, height = pil_image.size
    gaze_points = []

    for i in range(len(norm_bboxes)):
        xmin, ymin, xmax, ymax = norm_bboxes[i]
        color = colors[i % len(colors)]
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline=color, width=3)
        score = float(inout_scores[i]) if inout_scores is not None else 1.0

        if score > inout_thresh:
            heatmap = heatmaps[i].detach().cpu().numpy()
            max_index = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            gx = max_index[1] / heatmap.shape[1] * width
            gy = max_index[0] / heatmap.shape[0] * height
            cx = ((xmin + xmax) / 2) * width
            cy = ((ymin + ymax) / 2) * height
            draw.line([(cx, cy), (gx, gy)], fill=color, width=3)
            draw.ellipse([(gx-5, gy-5), (gx+5, gy+5)], fill=color)
            gaze_points.append((gx, gy))
        else:
            gaze_points.append(None)
    return overlay_image, gaze_points

# ---------------- Helper for multi-line text ----------------
def put_multiline_text(
    img,
    text,
    org,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.8,
    color=(255, 255, 255),
    thickness=2,
    line_spacing=30,
    max_width_ratio=0.85
):
    """
    Centered subtitle box with LEFT-aligned wrapped text.
    org = (center_x, bottom_y)
    """

    if not text.strip():
        return

    img_h, img_w = img.shape[:2]
    cx, cy = org
    max_width = int(img_w * max_width_ratio)

    words = text.split()
    lines = []
    current_line = ""

    # ---- Pixel-based wrapping ----
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        (w, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)

        if w <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    # ---- Measure block ----
    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_text_width = max(w for w, h in text_sizes)
    total_height = line_spacing * len(lines)

    # ---- Background box (centered) ----
    box_x1 = int(cx - max_text_width / 2) - 20
    box_y1 = int(cy - total_height) - 20
    box_x2 = int(cx + max_text_width / 2) + 20
    box_y2 = int(cy) + 10

    # Clamp
    box_x1 = max(10, box_x1)
    box_x2 = min(img_w - 10, box_x2)
    box_y1 = max(10, box_y1)
    box_y2 = min(img_h - 10, box_y2)

    overlay = img.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # ---- Draw LEFT-aligned text inside centered box ----
    text_x = box_x1 + 15
    y = box_y1 + line_spacing

    for line in lines:
        cv2.putText(img, line, (text_x, y), font, font_scale, color, thickness)
        y += line_spacing

# ---------------- Extract Audio & Transcribe ----------------
print("Extracting audio...")
clip = VideoFileClip(video_path)
audio_path = "temp_audio.wav"
clip.audio.write_audiofile(audio_path, fps=16000, verbose=False, logger=None)

print("Transcribing audio...")
segments_gen, _ = whisper_model.transcribe(audio_path, beam_size=5, word_timestamps=True)
segments=list(segments_gen)

# ---------------- Video Setup ----------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Total frames: {total_frames}, FPS: {fps}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(processed_video_path, fourcc, fps, (frame_width, frame_height))

# ---------------- Initialize Previous Predictions ----------------
prev_object_boxes, prev_object_names = [], []
prev_fixed_bboxes, prev_gaze_points = [], []
prev_dominant_emotions = []

frame_counter = 0
pbar = tqdm(desc="Processing frames", unit="frame")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for YOLO
    scale = 640 / frame.shape[1]
    small_frame = cv2.resize(frame, (640, int(frame.shape[0] * scale)))
    small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # ---- Only compute heavy models every 3rd frame ----
    if frame_counter % 3 == 0:
        # YOLO Object Detection (skip person)
        yolo_results = yolo(small_rgb, verbose=False)[0]
        object_boxes, object_names = [], []
        if hasattr(yolo_results, "boxes") and len(yolo_results.boxes) > 0:
            for box in yolo_results.boxes:
                cls_idx = int(box.cls[0].cpu().numpy())
                if cls_idx == 0:  # skip person
                    continue
                xyxy = box.xyxy[0].cpu().numpy() / scale
                label = yolo_results.names.get(cls_idx, str(cls_idx))
                object_boxes.append(xyxy)
                object_names.append(label)

        # YOLO Face Detection
        face_results = face_model(small_rgb, verbose=False)[0]
        fixed_bboxes = []
        if hasattr(face_results, "boxes") and len(face_results.boxes) > 0:
            for box in face_results.boxes:
                xyxy = box.xyxy[0].cpu().numpy() / scale
                fixed_bboxes.append(xyxy.astype(int))

        # Face recognition
        face_recognitions = []
        for (xmin, ymin, xmax, ymax) in fixed_bboxes:
            face_img = frame[ymin:ymax, xmin:xmax]
            detected = face_recognition(face_img)
            face_recognitions.append(detected["pred"])
            
            
        # Normalize for Gazelle
        H, W = frame.shape[:2]
        norm_bboxes = [[np.array([xmin/W, ymin/H, xmax/W, ymax/H]) for (xmin, ymin, xmax, ymax) in fixed_bboxes]]

        # Gazelle Inference
        if len(fixed_bboxes) > 0:
            img_tensor = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
            input_dict = {"images": img_tensor, "bboxes": norm_bboxes}
            with torch.no_grad():
                output = gazelle_model(input_dict)

            heatmaps = output["heatmap"][0]
            inout_scores = output["inout"][0] if output["inout"] is not None else None
            vis, gaze_points = visualize_gaze(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                                              heatmaps, norm_bboxes[0], inout_scores, inout_thresh)
        else:
            vis = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            gaze_points = []

        # Emotion detection
        dominant_emotions = []
        for (xmin, ymin, xmax, ymax) in fixed_bboxes:
            face_img = frame[ymin:ymax, xmin:xmax]
            dominant_emotion = "Neutral"
            try:
                emotions = emotion_detector.detect_emotions(face_img)
                if emotions:
                    dominant_emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
            except:
                pass
            dominant_emotions.append(dominant_emotion)

        # Save predictions for next frames
        prev_object_boxes = object_boxes
        prev_object_names = object_names
        prev_fixed_bboxes = fixed_bboxes
        prev_gaze_points = gaze_points
        prev_dominant_emotions = dominant_emotions
        prev_face_recognitions=face_recognitions

    else:
        # ---- Reuse previous predictions ----
        object_boxes = prev_object_boxes
        object_names = prev_object_names
        fixed_bboxes = prev_fixed_bboxes
        gaze_points = prev_gaze_points
        dominant_emotions = prev_dominant_emotions
        face_recognitions = prev_face_recognitions
        
        vis = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Convert PIL image back to numpy
    vis_np = cv2.cvtColor(np.array(vis), cv2.COLOR_RGBA2BGR)

    # ---- Overlay transcription for every frame ----
    current_time = frame_counter / fps
    display_words = []

    for seg in segments:
        if not hasattr(seg, "words") or not seg.words:
            continue

        for w in seg.words:
            if w.start <= current_time:
                display_words.append(w.word)
            else:
                break

    # Draw text only if something exists (limit to last 7 words)
    if display_words:
        text = " ".join(display_words[-7:])
        put_multiline_text(vis_np, text, (50, frame_height - 80))


    # ---- Draw emotions and face recognition results ----
    for (bbox, dom_em, identity) in zip(fixed_bboxes, dominant_emotions, face_recognitions):
       xmin, ymin, xmax, ymax = bbox

       # Draw emotion
       color = emotion_colors.get(dom_em.lower(), (255,255,255))
       cv2.putText(vis_np, dom_em.upper(), (xmin, max(15, ymin-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
       # Draw recognized identity (above the emotion)
       cv2.putText(vis_np, str(identity), (xmin, max(15, ymin-25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # green for identity
    # ---- Draw gaze objects ----
    for gp in gaze_points:
        if gp is not None:
            gx, gy = gp
        
        # Check if gaze is on any detected object
            gaze_on_object = False
            for obox, oname in zip(object_boxes, object_names):
                if point_in_box(gx, gy, obox):
                    label_text = f"Looking at: {oname}"
                    x1, y1, x2, y2 = obox.astype(int)
                    cv2.rectangle(vis_np, (x1, y1), (x2, y2), (0,255,0), 3)
                    gaze_on_object = True
                    break
        
        # If not on any object, determine where they're actually looking
            if not gaze_on_object:
            # Get face center
                if fixed_bboxes:  # Use first face
                    xmin, ymin, xmax, ymax = fixed_bboxes[0]
                    face_center_x = (xmin + xmax) / 2
                    face_center_y = (ymin + ymax) / 2
                
                # Calculate gaze direction relative to face
                    gaze_dx = gx - face_center_x
                    gaze_dy = gy - face_center_y
                
                # Check if gaze is pointing toward camera (approximation)
                # If gaze point is close to center of face, they're looking forward
                    gaze_distance = np.sqrt(gaze_dx**2 + gaze_dy**2)
                
                # Method 1: If gaze point is near face center (within 50 pixels)
                    if gaze_distance < 50:
                        label_text = "Looking at you"
                # Method 2: Check if gaze is pointing toward image center (camera)
                    elif abs(gx - frame_width/2) < 100 and abs(gy - frame_height/2) < 100:
                        label_text = "Looking at you"
                    else:
                        label_text = "Looking elsewhere"
                else:
                    label_text = "Gaze detected"
        else:
            label_text = "Looking at you"
    
        # Draw the text
        cv2.putText(vis_np, label_text, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0) if "Looking at:" in label_text else (0,0,255), 2)

    out.write(vis_np)
    frame_counter += 1
    pbar.update(1)

cap.release()
out.release()
cv2.destroyAllWindows()
pbar.close()
print(f"Processed video saved as {processed_video_path}")

# ---------------- Merge Audio with ffmpeg ----------------
print("Merging original audio into processed video using moviepy...")

# Load processed video (without audio)
processed_clip = VideoFileClip(processed_video_path)

# Load original audio from the original video
original_audio = VideoFileClip(video_path).audio

# Set audio to the processed video
final_clip = processed_clip.set_audio(original_audio)

# Write the final video with audio
final_clip.write_videofile(final_output_path, codec="libx264", audio_codec="aac")
print(f"Final video with audio saved as {final_output_path}")