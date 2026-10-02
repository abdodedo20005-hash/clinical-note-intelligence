"""TF-IDF + One-vs-Rest Logistic Regression classifier trained in the notebook."""
from pathlib import Path
from typing import List

import joblib
import numpy as np

from .constants import GROUPS, THRESHOLD


class LocalClassifier:
    def __init__(self, model_dir: Path):
        self.vectorizer = joblib.load(model_dir / "tfidf_vectorizer.joblib")
        self.classifier = joblib.load(model_dir / "tfidf_logistic_regression_classifier.joblib")
        if len(self.classifier.estimators_) != len(GROUPS):
            raise RuntimeError("Classifier output size does not match the GROUPS list.")

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        return self.classifier.predict_proba(self.vectorizer.transform(texts))

    def predict(self, texts: List[str]) -> List[dict]:
        """Groups with p >= threshold, ranked by probability. The top-1 group is always included."""
        results = []
        for p in self.predict_proba(texts):
            order = np.argsort(-p)
            keep = [int(i) for i in order if p[i] >= THRESHOLD] or [int(order[0])]
            results.append({
                "groups": [GROUPS[i] for i in keep],
                "confidence": float(p[order[0]]),
                "scores": {GROUPS[i]: round(float(p[i]), 4) for i in order},
            })
        return results
