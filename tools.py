import cv2
import os
import numpy as np
def to_gray(img):
    if img is None:
        return None
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    return gray


def load_error_map_for_image(image_path: str):
    base, _ = os.path.splitext(image_path)
    npy_path = base + "_err.npy"
    if os.path.exists(npy_path):
        return np.load(npy_path)
    return None