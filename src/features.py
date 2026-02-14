# TF-IDF
# Custom linguistic features
# Feature merging

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from .config import TFIDF_MAX_FEATURES

# reuse tokenizer/lemmatizer from preprocess to keep logic consistent
from .preprocess import tokenize, lemmatize_tokens

# hedges, source words, dates
HEDGE_WORDS = ["iddia", "söyleniyor", "öne sürüldü", "iddia edildi", "rapor edildi"]
SOURCE_WORDS = [
    "kaynak",
    "haber ajansı",
    "resmi açıklama",
    "bakanlık",
    "türkiye",
    "tdk",
]
DATE_PATTERN = re.compile(r"\b\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4}\b")


def extract_custom_features(texts):
    feats = []

    for raw in texts:
        # basic surface features
        caps = sum(1 for c in raw if c.isupper())
        cap_ratio = caps / (len(raw) + 1)

        exclam = raw.count("!")
        ex_ratio = exclam / (len(raw) + 1)

        q_ratio = raw.count("?") / (len(raw) + 1)

        hedge_flag = int(any(h in raw.lower() for h in HEDGE_WORDS))
        source_flag = int(any(s in raw.lower() for s in SOURCE_WORDS))
        link_flag = int("http" in raw.lower() or "www." in raw.lower())
        date_flag = int(bool(DATE_PATTERN.search(raw)))

        repeat_flag = int(bool(re.search(r"(.)\1{2,}", raw)))
        length = len(raw)

        # morphology-based features (best-effort via preprocess functions)
        tokens = tokenize(raw)
        lemmas = lemmatize_tokens(tokens)

        # lemma ratio: unique lemmas / tokens (higher => less repetition)
        lemma_ratio = len(set(lemmas)) / (len(tokens) + 1)

        # OOV ratio heuristic: tokens where lemma == token lowered (may indicate unknown)
        oov_ratio = sum(1 for t, l in zip(tokens, lemmas) if l == t.lower()) / (
            len(tokens) + 1
        )

        # simple POS proxies: count of tokens ending with -iyor, -di, etc (verb guesses)
        verb_guess = sum(
            1
            for t in tokens
            if t.endswith("iyor") or t.endswith("di") or t.endswith("mış")
        )
        verb_ratio = verb_guess / (len(tokens) + 1)

        # spelling-noise proxy: many repeated non-letter chars or long repeated vowels
        noise_score = 1.0 if re.search(r"(.)\1\1", raw) else 0.0

        feats.append(
            [
                cap_ratio,
                ex_ratio,
                q_ratio,
                hedge_flag,
                source_flag,
                link_flag,
                date_flag,
                repeat_flag,
                length,
                lemma_ratio,
                oov_ratio,
                verb_ratio,
                noise_score,
            ]
        )

    return np.array(feats)


def build_features(train_texts, test_texts):
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    X_train_text = vectorizer.fit_transform(train_texts).toarray()
    X_test_text = vectorizer.transform(test_texts).toarray()

    scaler = StandardScaler()

    X_train_custom = scaler.fit_transform(extract_custom_features(train_texts))
    X_test_custom = scaler.transform(extract_custom_features(test_texts))

    X_train = np.hstack([X_train_text, X_train_custom])
    X_test = np.hstack([X_test_text, X_test_custom])

    return X_train, X_test, vectorizer, scaler
