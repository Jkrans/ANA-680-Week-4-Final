from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "iris_model.pkl"


def load_data():
    data = load_iris()
    X = data.data
    y = data.target
    target_names = data.target_names
    feature_names = data.feature_names
    return X, y, target_names, feature_names


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    X, y, target_names, feature_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    cm = confusion_matrix(y_test, preds)

    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Save model + metadata (handy for the API)
    artifact = {
        "model": model,
        "target_names": target_names,
        "feature_names": feature_names,
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()