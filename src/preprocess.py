# Turkish normalization & cleaning.

import re
from typing import List

# Try Zemberek first (if installed)
_zemberek = None
_zeyrek = None

try:
    # zemberek-python (preferred if available)
    # API details vary between versions; we attempt common calls below.
    from zemberek import TurkishMorphology  # type: ignore

    _zemberek = TurkishMorphology  # assign class/namespace
except Exception:
    _zemberek = None

if _zemberek is None:
    try:
        # Zeyrek is a pure-Python fallback (lemmatizer + morphology)
        import zeyrek  # type: ignore
        from zeyrek import MorphAnalyzer  # type: ignore

        _zeyrek = MorphAnalyzer()
    except Exception:
        _zeyrek = None


def simple_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\sçğıöşü]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Try zemberek, then zeyrek, otherwise return tokens (lowercased) as lemmas.
    Implementation: best-effort; different libraries have different APIs.
    """
    if _zemberek:
        try:
            # Example: the zemberek-python API varies; adjust if needed.
            # This attempts single word analysis via a common API pattern.
            morph = (
                _zemberek.create_with_defaults()
                if hasattr(_zemberek, "create_with_defaults")
                else _zemberek()
            )
            lemmas = []
            for w in tokens:
                try:
                    analyses = (
                        morph.analyze_sentence(w)
                        if hasattr(morph, "analyze_sentence")
                        else morph.analyze(w)
                    )
                    # analyses structure varies; try to extract a lemma-safe field
                    # If the real API differs, inspect `analyses` and adapt
                    if analyses:
                        # best-effort extraction:
                        item = analyses[0]
                        # many wrappers return a `getLemmas()` method or a string
                        if hasattr(item, "getLemmas"):
                            l = item.getLemmas()
                            lemmas.append(l[0] if l else w.lower())
                        elif isinstance(item, str):
                            lemmas.append(item.lower())
                        else:
                            lemmas.append(str(item).lower())
                    else:
                        lemmas.append(w.lower())
                except Exception:
                    lemmas.append(w.lower())
            return lemmas
        except Exception:
            pass

    if _zeyrek:
        try:
            lemmas = []
            for w in tokens:
                res = (
                    _zeyrek.analyze(w)
                    if hasattr(_zeyrek, "analyze")
                    else _zeyrek.lemmatize(w)
                )
                # zeyrek.analyze returns list of parses, each parse like (surface, pos, lemma, feats)
                if res:
                    # pick first parse's lemma if present
                    first = res[0]
                    # different versions return different shapes — try to find lemma
                    if isinstance(first, tuple) and len(first) >= 3:
                        lemmas.append(first[2])
                    elif isinstance(first, dict) and "lemma" in first:
                        lemmas.append(first["lemma"])
                    else:
                        # some versions return list-of-lists
                        lemmas.append(str(first).lower())
                else:
                    lemmas.append(w.lower())
            return lemmas
        except Exception:
            pass

    # Fallback: lowercase tokens
    return [t.lower() for t in tokens]


def tokenize(text: str) -> List[str]:
    text = simple_clean(text)
    # naive tokenization (split on whitespace)
    return [t for t in text.split() if t]


def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline: cleaning + lemmatization (best-effort).
    Returns the lemmatized string (joined tokens).
    """
    tokens = tokenize(text)
    lemmas = lemmatize_tokens(tokens)
    return " ".join(lemmas)
