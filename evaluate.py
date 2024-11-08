import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
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


def plot_metrics_comparison(traditional_metrics, transformer_metrics):
    metrics = ["Precision", "Recall", "F1-Score"]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Bar chart comparison
    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width / 2, traditional_metrics[:3], width, label="Traditional (LR)")
    ax.bar(x + width / 2, transformer_metrics[:3], width, label="Transformer")

    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.close()


def create_confusion_matrices(X_test, y_test):
    # Load models and get predictions
    lr_model = joblib.load("models/logistic_regression.pkl")
    transformer = pipeline("text-classification", model="models/distilbert/")

    lr_pred = lr_model.predict(X_test)
    transformer_pred = [
        1 if pred["label"] == "LABEL_1" else 0 for pred in transformer(X_test.tolist())
    ]

    # Create confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Traditional model confusion matrix
    cm1 = confusion_matrix(y_test, lr_pred)
    sns.heatmap(cm1, annot=True, fmt="d", ax=ax1)
    ax1.set_title("Traditional (LR) Model\nConfusion Matrix")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    # Transformer model confusion matrix
    cm2 = confusion_matrix(y_test, transformer_pred)
    sns.heatmap(cm2, annot=True, fmt="d", ax=ax2)
    ax2.set_title("Transformer Model\nConfusion Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig("confusion_matrices.png")
    plt.close()


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

    # Create and save visualisations
    plot_metrics_comparison(traditional_metrics, transformer_metrics)
    create_confusion_matrices(X_test, y_test)


if __name__ == "__main__":
    main()
