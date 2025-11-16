# src/aggregator.py
from collections import Counter
import numpy as np

def majority_vote(strings):
    # strings: list of (text, conf)
    texts = [t for t,c in strings if t]
    if not texts:
        return "", 0.0
    cnt = Counter(texts)
    most, freq = cnt.most_common(1)[0]
    # compute average confidence for that string
    confs = [c for t,c in strings if t==most]
    avg_conf = np.mean(confs) if confs else 0.0
    return most, avg_conf, freq/len(strings)

def weighted_score(det_conf, ocr_conf, nlp_score):
    # simple weighted sum
    return 0.4*det_conf + 0.4*(ocr_conf or 0) + 0.2*nlp_score


