# face_embedder.py
"""
Minimal FaceNet embedding extractor:
- Load jpg/png/heic (+ EXIF transpose)
- (Optional) fix 90/180/270 rotation by testing 4 rotations and picking most upright face
- RetinaFace detect + eye-based alignment
- Square crop + resize
- FaceNet embedding (512-d, L2 normalized)

pip install facenet-pytorch retina-face opencv-python pillow pillow-heif matplotlib numpy torch torchvision
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from facenet_pytorch import InceptionResnetV1
from retinaface import RetinaFace


def load_image_bgr(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".heic", ".heif"]:
        import pillow_heif
        pillow_heif.register_heif_opener()

    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    rgb = np.array(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _largest(det: dict) -> float:
    x1, y1, x2, y2 = det["facial_area"]
    return float((x2 - x1) * (y2 - y1))


def _rot90(img: np.ndarray, k: int) -> np.ndarray:
    if k % 4 == 0:
        return img
    return np.ascontiguousarray(np.rot90(img, k))


def _upright_score(det: dict) -> float:
    """
    Higher is better.
    Encourages: eyes above mouth, eyes mostly horizontal.
    """
    lm = det.get("landmarks", None)
    if lm is None:
        return -1e9

    le = np.array(lm["left_eye"], dtype=np.float32)
    re = np.array(lm["right_eye"], dtype=np.float32)
    ml = np.array(lm["mouth_left"], dtype=np.float32)
    mr = np.array(lm["mouth_right"], dtype=np.float32)

    # ensure left/right by x in current image
    if re[0] < le[0]:
        le, re = re, le
    if mr[0] < ml[0]:
        ml, mr = mr, ml

    eye_y = float((le[1] + re[1]) * 0.5)
    mouth_y = float((ml[1] + mr[1]) * 0.5)

    dx = abs(float(re[0] - le[0]))
    dy = abs(float(re[1] - le[1]))

    score = float(det.get("score", 0.0))
    if eye_y < mouth_y:
        score += 2.0
    else:
        score -= 3.0

    # prefer eyes more horizontal than vertical
    score += 1.0 if dx >= dy else -1.0
    return score


def _best_upright_rotation(img_bgr: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Try 0/90/180/270 rotations and pick the rotation with best upright_score.
    Returns (best_img, best_det_for_that_img).
    """
    best_img = img_bgr
    best_det = None
    best_score = -1e9

    for k in (0, 1, 2, 3):
        cand = _rot90(img_bgr, k)
        dets = RetinaFace.detect_faces(cand)
        if not isinstance(dets, dict) or len(dets) == 0:
            continue

        det = max(dets.values(), key=_largest)
        sc = _upright_score(det)
        if sc > best_score:
            best_score = sc
            best_img = cand
            best_det = det

    if best_det is None:
        raise ValueError("No face detected in any 90-degree rotation.")
    return best_img, best_det


def _crop_square(img: np.ndarray, bbox: Tuple[int, int, int, int], margin: float) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    side = int(max(bw, bh) * (1.0 + margin))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    x1n, y1n = cx - side // 2, cy - side // 2
    x2n, y2n = x1n + side, y1n + side

    pad_l = max(0, -x1n)
    pad_t = max(0, -y1n)
    pad_r = max(0, x2n - w)
    pad_b = max(0, y2n - h)

    if pad_l or pad_t or pad_r or pad_b:
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        x1n += pad_l; x2n += pad_l
        y1n += pad_t; y2n += pad_t

    return img[y1n:y2n, x1n:x2n]


class FaceNetRetinaEmbedder:
    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        pretrained: str = "vggface2",
        output_size: int = 160,
        margin: float = 0.35,
        fix_upright_rot90: bool = True,
    ):
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_size = int(output_size)
        self.margin = float(margin)
        self.fix_upright_rot90 = bool(fix_upright_rot90)
        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(self.device)

    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        # 1) make image upright first (fix 90/180/270) if enabled
        if self.fix_upright_rot90:
            img_bgr, det = _best_upright_rotation(img_bgr)
        else:
            dets = RetinaFace.detect_faces(img_bgr)
            if not isinstance(dets, dict) or len(dets) == 0:
                raise ValueError("No face detected.")
            det = max(dets.values(), key=_largest)

        x1, y1, x2, y2 = map(int, det["facial_area"])
        lm = det["landmarks"]

        le = np.array(lm["left_eye"], dtype=np.float32)
        re = np.array(lm["right_eye"], dtype=np.float32)
        if re[0] < le[0]:
            le, re = re, le

        # 2) rotate to level eyes
        dx = float(re[0] - le[0])
        dy = float(re[1] - le[1])
        angle = np.degrees(np.arctan2(dy, dx))  # y-down coords

        center = ((le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0)
        h, w = img_bgr.shape[:2]
        M = cv2.getRotationMatrix2D(center, angle, 1.0)  # rotate clockwise to level if needed
        rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

        # 3) rotate bbox corners -> new bbox
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        pts_h = np.hstack([pts, np.ones((4, 1), dtype=np.float32)])
        rpts = (M @ pts_h.T).T
        rx1, ry1 = int(np.floor(rpts[:, 0].min())), int(np.floor(rpts[:, 1].min()))
        rx2, ry2 = int(np.ceil(rpts[:, 0].max())), int(np.ceil(rpts[:, 1].max()))

        # 4) crop + resize
        crop = _crop_square(rotated, (rx1, ry1, rx2, ry2), margin=self.margin)
        crop = cv2.resize(crop, (self.output_size, self.output_size), interpolation=cv2.INTER_AREA)
        return crop

    @staticmethod
    def _to_facenet_tensor(face_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

    @torch.no_grad()
    def embed(self, img_path: str, show_debug: bool = False) -> Optional[torch.Tensor]:
        img_bgr = load_image_bgr(img_path)
        try:
            face = self.preprocess(img_bgr)
        except Exception:
            return None

        if show_debug:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1); plt.title("Original"); plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); plt.axis("off")
            plt.subplot(1, 2, 2); plt.title("Aligned Crop"); plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)); plt.axis("off")
            plt.tight_layout(); plt.show()

        x = self._to_facenet_tensor(face).to(self.device)
        emb = self.model(x).squeeze(0)
        return torch.nn.functional.normalize(emb, p=2, dim=0)
