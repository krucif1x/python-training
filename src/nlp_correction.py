# src/nlp_correct.py
import re
import Levenshtein
from difflib import get_close_matches

# Basic replacements for typical OCR confusions
REPLACEMENTS = {
    '0':'0', 'O':'0', 'o':'0',
    '1':'1', 'I':'1', 'l':'1',
    '5':'5','S':'5',
    '8':'8','B':'8',
    '2':'2','Z':'2',
}

def normalize_text(raw):
    # remove non-alnum, uppercase, convert spaces to single
    s = re.sub(r'[^A-Za-z0-9]', '', raw).upper()
    return s

def fix_common_confusions(s):
    out = []
    for ch in s:
        out.append(REPLACEMENTS.get(ch, ch))
    return ''.join(out)

# Example regex pattern for Indonesian: prefix letters (1-2) + 1-4 digits + suffix letters (1-3)
PLATE_REGEX = re.compile(r'^([A-Z]{1,2})(\d{1,4})([A-Z]{0,3})$')

# Known prefixes (small sample; expand with real data)
KNOWN_PREFIXES = ['B','D','L','DK','F','AB','AD']  # add more

def pattern_score(s):
    if PLATE_REGEX.match(s):
        return 1.0
    return 0.0

def best_prefix_match(prefix):
    matches = get_close_matches(prefix, KNOWN_PREFIXES, n=1, cutoff=0.6)
    return matches[0] if matches else prefix

def nlp_postprocess(raw_text):
    s = normalize_text(raw_text)
    s = fix_common_confusions(s)

    m = PLATE_REGEX.match(s)
    if m:
        prefix, nums, suffix = m.groups()
        prefix = best_prefix_match(prefix)
        formatted = f"{prefix} {nums} {suffix}".strip()
        score = 1.0
        return formatted, score

    # If not matched, try corrections:
    # Try sliding insertions of spaces: check splits where left is letters, middle digits, right letters
    for i in range(1,4):
        for j in range(i+1, i+6):
            a = s[:i]; b = s[i:j]; c = s[j:]
            if a.isalpha() and b.isdigit():
                cand = f"{a} {b} {c}"
                cand_s = cand.replace(" ", "")
                if PLATE_REGEX.match(cand_s):
                    prefix = best_prefix_match(a)
                    formatted = f"{prefix} {b} {c}"
                    return formatted, 0.9

    # fallback: return cleaned but flagged
    return s, 0.2
