# Encapsulates:
# Model creation
# Train
# Predict
# Confidence score

# This keeps train.py clean.

import numpy as np
import pickle
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from scipy.special import expit
from .config import *


class TsetlinModel:

    def __init__(self):
        self.model = MultiClassTsetlinMachine(
            number_of_clauses=TSETLIN_CLAUSES, T=TSETLIN_T, s=TSETLIN_S
        )

    def fit(self, X, y):
        X_bin = (X > 0).astype(int)
        self.model.fit(X_bin, y, epochs=EPOCHS)

    def predict(self, X):
        X_bin = (X > 0).astype(int)
        return self.model.predict(X_bin)

    def confidence(self, X):
        X_bin = (X > 0).astype(int)
        votes = self.model.transform(X_bin)
        return expit(votes.mean(axis=1))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
