# src/preprocess.py
import cv2
import numpy as np


def crop_and_warp(img, box, pad=5):
    x1,y1,x2,y2 = box
    h,w = img.shape[:2]
    x1 = max(0,x1-pad); y1 = max(0,y1-pad)
    x2 = min(w,x2+pad); y2 = min(h,y2+pad)
    crop = img[y1:y2, x1:x2]
    return crop

def enhance_for_ocr(img):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)
    # Bilateral filter to preserve edges
    filt = cv2.bilateralFilter(cl, 9, 75, 75)
    # Resize to bigger so OCR can work
    h,w = filt.shape
    scale = 2
    resized = cv2.resize(filt, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    # Adaptive threshold
    th = cv2.adaptiveThreshold(resized,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,11,2)
    return th

def deskew_and_rectify(image):
    # Placeholder: if plate is skewed use cv2.getPerspectiveTransform after detecting corners
    return image
