# Models Directory

Currently, the `.pkl` files in this folder are **empty placeholders**.
The project is **not using any saved models yet**.

---

## Current Model (What the project is using now)

Right now, the system runs with a **simple baseline pipeline**:

- Text → TF-IDF (built in memory each run)
- Handcrafted linguistic features
- Fuzzy rule-based fake score
- Tsetlin Machine trained during runtime (not loaded from disk)

Since nothing is loaded from `.pkl` files, the model is **retrained every time** and no persistent state exists.

---

## What Each `.pkl` File Should Contain

To make the system persistent and stronger, these files should store trained objects:

### `tfidf.pkl`

Should contain a fitted **TF-IDF Vectorizer**:

- Vocabulary
- IDF weights
- Tokenization settings

Purpose: Convert text to consistent numeric features.

---

### `scaler.pkl`

Should contain a trained **feature scaler**:

- Normalization parameters for handcrafted features

Purpose: Keep feature ranges stable across runs.

---

### `tsetlin_model.pkl`

Should contain the **trained Tsetlin Machine**:

- Learned clauses
- Automata states
- Classification logic

Purpose: Provide REAL vs FAKE prediction without retraining.

---

## Why This Matters

Saving trained models allows:

- Consistent predictions
- Faster startup (no retraining)
- Better performance after proper training
- Reproducible results

---

## How to Improve the System

To improve accuracy and stability, the `.pkl` files should store:

- A properly trained TF-IDF on the full dataset
- A fitted scaler for all numeric features
- A well-trained Tsetlin Machine (more epochs, tuned parameters)

Optional future upgrades:

- Better feature engineering
- Stronger classifier
- Semantic embeddings

---
