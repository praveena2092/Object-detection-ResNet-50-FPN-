import cv2
import numpy as np


def enhance_low_light(img_bgr, brightness_thresh=80, clip_limit=2.5, gamma=1.6):
    """CLAHE on the L channel + gamma correction, applied only to dark images."""
    mean_v = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 2].mean()
    if mean_v >= brightness_thresh:
        return img_bgr
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(l)
    img_bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    g = gamma if mean_v < 50 else gamma * 0.7
    table = np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img_bgr, table)
