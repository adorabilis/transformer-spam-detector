from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from utils.data_loader import load_data


def train():
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "logistic_regression.pkl"

    X_train, X_test, y_train, y_test = load_data()

    model = make_pipeline(
        TfidfVectorizer(max_features=1000), LogisticRegression(class_weight="balanced")
    )

    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print("Logistic regression model trained.")


if __name__ == "__main__":
    train()
