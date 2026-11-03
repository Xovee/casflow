"""Historical CasFlow reporting metrics (unchanged from public TF code)."""

import numpy as np


def report(predictions, labels):
    predictions = np.maximum(np.asarray(predictions, dtype=np.float64), 1)
    labels = np.maximum(np.asarray(labels, dtype=np.float64), 1)
    return {
        "msle": float(np.mean((np.log2(predictions) - np.log2(labels)) ** 2)),
        "mape": float(np.mean(np.abs(np.log2(predictions + 1) - np.log2(labels + 1)) / np.log2(labels + 2))),
    }
