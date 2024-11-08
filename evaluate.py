import joblib
from sklearn.metrics import precision_recall_fscore_support
from transformers import pipeline

from utils.data_loader import load_data


def evaluate_traditional_model(X_test, y_test):
    model = joblib.load("models/logistic_regression.pkl")
    y_pred = model.predict(X_test)
    return precision_recall_fscore_support(y_test, y_pred, average="binary")


def evaluate_transformer_model(X_test, y_test):
    classifier = pipeline("text-classification", model="models/distilbert/")
    predictions = classifier(X_test.tolist())
    y_pred = [1 if pred["label"] == "LABEL_1" else 0 for pred in predictions]
    return precision_recall_fscore_support(y_test, y_pred, average="binary")


def main():
    _, X_test, _, y_test = load_data()

    traditional_metrics = evaluate_traditional_model(X_test, y_test)
    transformer_metrics = evaluate_transformer_model(X_test, y_test)

    print("\nLR Model Metrics:")
    print(f"Precision: {traditional_metrics[0]:.3f}")
    print(f"Recall: {traditional_metrics[1]:.3f}")
    print(f"F1-Score: {traditional_metrics[2]:.3f}")

    print("\nTransformer Model Metrics:")
    print(f"Precision: {transformer_metrics[0]:.3f}")
    print(f"Recall: {transformer_metrics[1]:.3f}")
    print(f"F1-Score: {transformer_metrics[2]:.3f}")


if __name__ == "__main__":
    main()
