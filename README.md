# Turkish Fake News Detection (Tsetlin Prototype)

## Overview

A Turkish fake news detector combining:

- **TF-IDF vectorization** for text representation
- **Custom linguistic features** (capitalization, punctuation, hedge words, evidence markers)
- **Fuzzy logic system** for sensationalism and credibility scoring
- **Tsetlin Machine** for binary classification (REAL/FAKE)

## Current Status

✅ **Implemented:**

- Text preprocessing with optional Turkish morphology (Zemberek/Zeyrek)
- Feature engineering pipeline (TF-IDF + custom + fuzzy)
- Model training on raw datasets
- Single text prediction with confidence scores
- Test prediction suite

⚠️ **Limitations:**

- Model retrains on every run (no persistence)
- Small handcrafted feature set
- Fuzzy rules are manually tuned

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train on dataset
python main.py --train data/raw

# Predict single text
python main.py --predict "SON DAKİKA mucize ilaç bulundu!!!"

# Run test predictions
python -m tools.test_predict
```

## Future Improvements

🎯 **Planned:**

- Save trained models (TF-IDF, scaler, Tsetlin) to disk for faster inference
- Expand dataset and improve fuzzy rule tuning
- Add semantic embeddings (Turkish BERT/FastText)
- Implement cross-validation for robust evaluation
- Add explainability features (top influential words, clause activation)
