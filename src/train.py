# load >> preprocess >> feature >> train >> save model

import re
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from .data_loader import load_data
from .preprocess import preprocess
from .features import build_features
from .model_tsetlin import TsetlinModel
from .config import *


def train_pipeline(dataset_path):

    print("Loading data...")
    df = load_data(dataset_path)

    df["clean"] = df["text"].apply(preprocess)

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("Building features...")
    X_train, X_test, vectorizer, scaler = build_features(X_train_txt, X_test_txt)

    from .fuzzy import compute_fuzzy_score

    # compute fuzzy for each sample (train & test) using the same logic used in features.py
    def build_fuzzy_array(texts):
        arr = []
        for t in texts:
            # derive the same component signals used in fuzzy:
            # sensationalism ~ cap_ratio + exclam + repeat
            s = (sum(1 for c in t if c.isupper()) / (len(t) + 1)) + (
                t.count("!") / (len(t) + 1)
            )
            s += 1.0 if re.search(r"(.)\1{2,}", t) else 0.0
            # evidence ~ source + link + date
            e = (
                int(
                    any(
                        skw in t.lower()
                        for skw in [
                            "kaynak",
                            "haber ajansı",
                            "resmi açıklama",
                            "bakanlık",
                        ]
                    )
                )
                + int("http" in t.lower() or "www." in t.lower())
                + (1 if re.search(r"\b\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4}\b", t) else 0)
            )
            # hedge ~ hedges
            h = int(
                any(
                    hd in t.lower()
                    for hd in ["iddia", "söyleniyor", "iddia edildi", "öne sürüldü"]
                )
            )
            # noise ~ repeated char or weird tokens
            n = 1.0 if re.search(r"(.)\1\1", t) else 0.0

            score = compute_fuzzy_score(
                {"sensationalism": s, "evidence": e, "hedge": h, "noise": n}
            )
            arr.append([score])
        return np.array(arr)

    X_train_fuzzy = build_fuzzy_array(X_train_txt)
    X_test_fuzzy = build_fuzzy_array(X_test_txt)

    # append fuzzy score as extra column to features
    X_train = np.hstack([X_train, X_train_fuzzy])
    X_test = np.hstack([X_test, X_test_fuzzy])

    print("Training Tsetlin Machine...")
    tm = TsetlinModel()
    tm.fit(X_train, y_train)

    preds = tm.predict(X_test)
    conf = tm.confidence(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nConfidence Scores:")
    print(conf)

    print("Saving model...")
    tm.save(MODEL_PATH)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print("Training complete.")
