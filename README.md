# IRIS HUD - Laiba Asmatullah, Madhav Gupta
## Demo


https://github.com/user-attachments/assets/5f786656-9c56-43db-b01d-9b1c69bc2bc0

## Project Overview

This project presents a **Smart Glasses System** designed to understand user attention and surroundings using computer vision and AI techniques. The system combines multiple perception modules to analyze faces, emotions, gaze direction, objects in view, and spoken audio.

The system can operate in two modes:

* **Video-based processing**
* **Live camera processing**


---

## Features

* **Face Recognition** – Identifies known individuals using trained face IDs
* **Emotion Recognition** – Detects facial emotions
* **Gaze Detection** – Estimates gaze direction (video mode only)
* **Object Detection** – Detects objects in the environment
* **Attention Tracking** – Checks whether the user’s gaze is focused on a detected object
* **Live Transcription** – Converts speech to text during live operation

---

## Project Structure

```
project-root/
│
├── source/
│   ├── attention_tracking.py   # Runs detection on video input
│   ├── live_code.py            # Runs detection on live camera using OpenCV
│   ├── face.py              # Contains funtions for generating and checking face embeddings agaist the stored embeddings.
├   |── YoloObjTraining/  # Object detection training models and scripts
|   |── faceid/          # Face recognition (face IDs) training models and script
|   |── models/          # This folder contains weights for yolo v8 for face and object detection. Please download this folder from the drive link:
                         https://drive.google.com/drive/folders/1n0wtizOewng-qIcrYg5KiqYsYKvEld6F
│
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\\Scripts\\activate      # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the System

All executable scripts are located in the **`source/`** directory.

### Video Mode (Full Functionality)

```bash
python source/attention_tracking.py
```

**MUST UPDATE PATHS IN ATTECTION_TRACKING.PY**

**Supported features:**

* Face recognition
* Emotion recognition
* Object detection
* Gaze detection
* Attention tracking

This mode is recommended for complete system evaluation.

---

### Live Camera Mode

```bash
python source/live_code.py
```

**Supported features:**

* Face recognition
* Emotion recognition
* Object detection
* Live transcription

**Note:** Gaze detection is not available in live mode due to the model’s computational cost.

---

## Notes & Troubleshooting

* If you encounter file or model loading errors, you may need to **update file paths** inside the code.
* Ensure the required model files are present in their respective directories.

---

## Training Directories

* **Object Detection Training Folder**

  * Contains scripts and models used to train the object detection module

* **Face IDs Training Folder**

  * Contains data and models for face recognition training


---

## License

This project is intended for academic and research purposes.
