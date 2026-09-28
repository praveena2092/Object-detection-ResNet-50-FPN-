import glob
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from .enhance import enhance_low_light


def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    """Normalized YOLO [xc, yc, w, h] -> absolute [x1, y1, x2, y2]."""
    return [(xc - w / 2) * img_w, (yc - h / 2) * img_h,
            (xc + w / 2) * img_w, (yc + h / 2) * img_h]


def list_labeled_images(img_dir: str, lbl_dir: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(img_dir, "*")))
    return [p for p in paths if os.path.exists(os.path.join(lbl_dir, Path(p).stem + ".txt"))]


def split_train_val(paths: List[str], val_fraction: float, seed: int) -> Tuple[List[str], List[str]]:
    paths = list(paths)
    random.Random(seed).shuffle(paths)
    n_val = int(len(paths) * val_fraction)
    return paths[n_val:], paths[:n_val]


def collate_fn(batch):
    return tuple(zip(*batch))


class MosquitoDataset(Dataset):
    """Returns (image_tensor, target, path). Labels are shifted +1 (0 = background)."""

    def __init__(self, image_paths: List[str], label_dir: Optional[str],
                 enhance: bool = True, train: bool = False, hflip: bool = True):
        self.image_paths = image_paths
        self.label_dir = label_dir
        self.enhance = enhance
        self.train = train
        self.hflip = hflip

    def __len__(self):
        return len(self.image_paths)

    def _read_boxes(self, stem, img_w, img_h):
        boxes, labels = [], []
        path = os.path.join(self.label_dir, stem + ".txt") if self.label_dir else None
        if path and os.path.exists(path):
            with open(path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls_id, xc, yc, w, h = map(float, parts[:5])
                    x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, img_w, img_h)
                    if x2 <= x1 or y2 <= y1:  # skip degenerate boxes
                        continue
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cls_id) + 1)
        return boxes, labels

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise IOError(f"Could not read image: {img_path}")
        if self.enhance:
            img = enhance_low_light(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        boxes, labels = self._read_boxes(Path(img_path).stem, img_w, img_h)
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        img_t = F.to_tensor(img)
        if self.train and self.hflip and random.random() < 0.5:
            img_t = img_t.flip(-1)
            if len(boxes_t):
                boxes_t = torch.stack([img_w - boxes_t[:, 2], boxes_t[:, 1],
                                       img_w - boxes_t[:, 0], boxes_t[:, 3]], dim=1)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "area": (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1]),
            "iscrowd": torch.zeros((len(boxes_t),), dtype=torch.int64),
        }
        return img_t, target, img_path
