"""Build and validate the competition submission.

Format: id, ImageID, LabelName, Conf, xcenter, ycenter, bbx_width, bbx_height
(one row per image, rows in the same order as sample_submission.csv).
"""
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import CLASSES
from .dataset import MosquitoDataset

COLUMNS = ["id", "ImageID", "LabelName", "Conf", "xcenter", "ycenter", "bbx_width", "bbx_height"]


@torch.no_grad()
def predict_test(model, test_images: List[str], device, score_thresh: float, enhance: bool = True) -> pd.DataFrame:
    ds = MosquitoDataset(test_images, label_dir=None, enhance=enhance, train=False)
    rows = []
    model.eval()
    for i in tqdm(range(len(ds)), desc="Predicting test set"):
        img, _, path = ds[i]
        _, h, w = img.shape
        pred = model([img.to(device)])[0]
        boxes, scores, labels = (pred[k].cpu().numpy() for k in ("boxes", "scores", "labels"))
        keep = scores >= score_thresh
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        row = {"ImageID": Path(path).name, "class_id": None, "Conf": None,
               "xcenter": None, "ycenter": None, "bbx_width": None, "bbx_height": None}
        if len(boxes):
            b = int(np.argmax(scores))  # keep the top detection per image
            x1, y1, x2, y2 = boxes[b]
            row.update(class_id=int(labels[b]) - 1, Conf=float(scores[b]),
                       xcenter=((x1 + x2) / 2) / w, ycenter=((y1 + y2) / 2) / h,
                       bbx_width=(x2 - x1) / w, bbx_height=(y2 - y1) / h)
        rows.append(row)
    return pd.DataFrame(rows)


def finalize_submission(pred_df: pd.DataFrame, sample_sub_path: str) -> pd.DataFrame:
    """Fill empty detections with a low-confidence default and match sample row order."""
    df = pred_df.copy()
    df["class_id"] = df["class_id"].fillna(0).astype(int)
    df["Conf"] = df["Conf"].fillna(0.0)
    df[["xcenter", "ycenter"]] = df[["xcenter", "ycenter"]].fillna(0.5)
    df[["bbx_width", "bbx_height"]] = df[["bbx_width", "bbx_height"]].fillna(0.1)
    df["LabelName"] = df["class_id"].map(dict(enumerate(CLASSES)))

    sample = pd.read_csv(sample_sub_path)
    df = df.set_index("ImageID").loc[sample["ImageID"]].reset_index()
    df["id"] = range(len(df))
    return df[COLUMNS]


def validate_submission(sub: pd.DataFrame, sample_sub_path: str) -> List[str]:
    """Return a list of problems (empty list = looks valid)."""
    sample = pd.read_csv(sample_sub_path)
    problems = []
    if list(sub.columns) != list(sample.columns):
        problems.append(f"columns differ: expected {list(sample.columns)}, got {list(sub.columns)}")
    if len(sub) != len(sample):
        problems.append(f"row count differs: expected {len(sample)}, got {len(sub)}")
    elif "ImageID" in sub and not (sub["ImageID"].values == sample["ImageID"].values).all():
        problems.append("ImageID order does not match sample_submission.csv")
    bad = set(sub["LabelName"]) - set(CLASSES) if "LabelName" in sub else set()
    if bad:
        problems.append(f"unexpected LabelName values: {sorted(bad)}")
    for c in ("xcenter", "ycenter", "bbx_width", "bbx_height"):
        if c in sub and ((sub[c] < 0) | (sub[c] > 1)).any():
            problems.append(f"{c} has values outside [0, 1]")
    return problems
