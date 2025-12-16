# augmentation.py
"""
Minimal offline augmentation writer for TRAIN folders.

Applies (independently with probabilities):
- horizontal flip
- brightness
- contrast
- grayscale

Writes new JPGs beside originals:
  name_flipped.jpg, name_bright.jpg, name_contrast.jpg, name_gray.jpg

pip install pillow pillow-heif
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageEnhance, ImageOps

# heic support (optional but recommended)
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except Exception:
    pass

IMG_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


@dataclass
class AugmentConfig:
    flip_prob: float = 0.70
    brightness_prob: float = 0.10
    contrast_prob: float = 0.10
    grayscale_prob: float = 0.05
    brightness_range: Tuple[float, float] = (0.75, 1.25)
    contrast_range: Tuple[float, float] = (0.75, 1.25)
    jpeg_quality: int = 95


def augment_training_folder(train_dir: str | Path, cfg: AugmentConfig, seed: int = 42, verbose: bool = True):
    train_dir = Path(train_dir)
    rng = random.Random(seed)

    counts = {"flipped": 0, "bright": 0, "contrast": 0, "gray": 0, "skipped": 0}

    for person_dir in sorted([p for p in train_dir.iterdir() if p.is_dir()]):
        for img_path in sorted(person_dir.iterdir()):
            if not (img_path.is_file() and img_path.suffix.lower() in IMG_EXTS):
                continue

            stem = img_path.stem.lower()
            if stem.endswith(("_flipped", "_bright", "_contrast", "_gray")):
                counts["skipped"] += 1
                continue

            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img).convert("RGB")

            base = img_path.stem

            if rng.random() < cfg.flip_prob:
                out = person_dir / f"{base}_flipped.jpg"
                if not out.exists():
                    img.transpose(Image.FLIP_LEFT_RIGHT).save(out, "JPEG", quality=cfg.jpeg_quality, optimize=True)
                    counts["flipped"] += 1

            if rng.random() < cfg.brightness_prob:
                out = person_dir / f"{base}_bright.jpg"
                if not out.exists():
                    f = rng.uniform(*cfg.brightness_range)
                    ImageEnhance.Brightness(img).enhance(f).save(out, "JPEG", quality=cfg.jpeg_quality, optimize=True)
                    counts["bright"] += 1

            if rng.random() < cfg.contrast_prob:
                out = person_dir / f"{base}_contrast.jpg"
                if not out.exists():
                    f = rng.uniform(*cfg.contrast_range)
                    ImageEnhance.Contrast(img).enhance(f).save(out, "JPEG", quality=cfg.jpeg_quality, optimize=True)
                    counts["contrast"] += 1

            if rng.random() < cfg.grayscale_prob:
                out = person_dir / f"{base}_gray.jpg"
                if not out.exists():
                    img.convert("L").convert("RGB").save(out, "JPEG", quality=cfg.jpeg_quality, optimize=True)
                    counts["gray"] += 1

    if verbose:
        print(counts)
    return counts
