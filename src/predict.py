import re
import pickle
import numpy as np

from .preprocess import preprocess
from .model_tsetlin import TsetlinModel
from .features import extract_custom_features
from .fuzzy import compute_fuzzy_score
from .config import *


def _clamp01(x):
    return max(0.0, min(1.0, float(x)))


def _extract_fuzzy_inputs(text: str):
    """
    produce normalized fuzzy inputs in range [0,1].
      - sensationalism >> more fake
      - evidence  >> more real  (so we invert later)
      - hedge  >> more fake
      - noise  >> more fake
    """

    lower = text.lower()
    length = max(len(text), 1)

    # sensationalism
    upper_ratio = sum(1 for c in text if c.isupper()) / length
    exclam_ratio = text.count("!") / length
    repeat_flag = 1.0 if re.search(r"(.)\1\1", text) else 0.0

    sensationalism = _clamp01(upper_ratio * 2.5 + exclam_ratio * 5 + repeat_flag * 0.6)

    # evidence -- REAL indicator
    evidence_keywords = [
        "kaynak",
        "haber ajansı",
        "resmi açıklama",
        "bakanlık",
        "verilere göre",
        "rapora göre",
        "araştırmaya göre",
    ]

    evidence = 0.0
    if any(k in lower for k in evidence_keywords):
        evidence += 0.5
    if "http" in lower or "www." in lower:
        evidence += 0.3
    if re.search(r"\b\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4}\b", text):
        evidence += 0.2

    evidence = _clamp01(evidence)

    # hedge -- uncertainty language
    hedge_keywords = [
        "iddia",
        "iddia edildi",
        "söyleniyor",
        "öne sürüldü",
        "iddialara göre",
        "iddia ediliyor",
    ]

    hedge = 1.0 if any(h in lower for h in hedge_keywords) else 0.0

    noise = 1.0 if re.search(r"(.)\1\1", text) else 0.0

    # evidence means REAL >> convert to "fake evidence lack"
    evidence_for_fake = 1.0 - evidence

    return {
        "sensationalism": sensationalism,
        "evidence": evidence_for_fake,
        "hedge": hedge,
        "noise": noise,
    }


def predict_text(text):
    """
    predict single Turkish text.
    outputs:
      - TM confidence (0..1)
      - Fuzzy fake_score (0..1)
    """

    # load artifacts
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    tm = TsetlinModel()
    tm.load(MODEL_PATH)

    # build feature vector
    clean = preprocess(text)
    X_text = vectorizer.transform([clean]).toarray()
    X_custom = scaler.transform(extract_custom_features([text]))

    X = np.hstack([X_text, X_custom])

    # fuzzy
    fuzzy_inputs = _extract_fuzzy_inputs(text)

    print("\n[FUZZY DEBUG]")
    for k, v in fuzzy_inputs.items():
        print(f"  {k:<15} = {v:.3f}")

    fs = compute_fuzzy_score(fuzzy_inputs)

    # append fuzzy feature
    X = np.hstack([X, np.array([[fs]])])

    # prediction
    pred = tm.predict(X)[0]
    conf = tm.confidence(X)[0]

    # determine likelihood based on fuzzy score
    fuzzy_likelihood = "likely FAKE" if fs >= 0.5 else "likely REAL"

    print("\n" + "-" * 50)
    print("PREDICTION SCORES")
    print("-" * 50)
    print(f"Fuzzy Score:             {fs:.3f}  [{fuzzy_likelihood}]")
    print(f"Model Confidence:        {conf:.3f}")

    print("-" * 50)

    return {
        "tm_confidence": float(conf),
        "fuzzy_score": float(fs),
    }
