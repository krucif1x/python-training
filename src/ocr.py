# src/ocr_engine.py
import easyocr
import pytesseract
import numpy as np

reader = easyocr.Reader(['en'])  # add 'id' if supported

def ocr_easyocr(img):
    # img: grayscale or BGR
    results = reader.readtext(img)
    # results: list of (bbox, text, confidence)
    texts = []
    for bbox, text, conf in results:
        texts.append({"text": text, "conf": float(conf)})
    # sometimes it returns multiple chunks; join by x-coordinate
    if not texts:
        return "", 0.0
    # pick best by confidence or join
    texts_sorted = sorted(texts, key=lambda x: x['conf'], reverse=True)
    return texts_sorted[0]['text'], texts_sorted[0]['conf']

def ocr_tesseract(img):
    # config tweak
    config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(img, config=config)
    # rudimentary confidence: tesseract returns confidences in TSV but for speed skip
    return text.strip(), None

def ocr_pipeline(img):
    # img: preprocessed threshold image (numpy)
    text, conf = ocr_easyocr(img)
    if conf < 0.4 or text == "":
        text2, conf2 = ocr_tesseract(img)
        if text2:
            return text2, conf2 or 0.3
    return text, conf
