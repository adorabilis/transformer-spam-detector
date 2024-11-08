import os

from datasets import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)

from utils.data_loader import load_data


def train():
    # Create models directory if it doesn't exist
    model_path = "models/distilbert/"
    os.makedirs(model_path, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()

    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Tokenize the data
    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True)

    # Create datasets with tokenized inputs
    train_dataset = Dataset.from_dict(
        {
            "input_ids": train_encodings["input_ids"],
            "attention_mask": train_encodings["attention_mask"],
            "label": y_train.tolist(),
        }
    )

    test_dataset = Dataset.from_dict(
        {
            "input_ids": test_encodings["input_ids"],
            "attention_mask": test_encodings["attention_mask"],
            "label": y_test.tolist(),
        }
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased"
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

    # Save both model and tokenizer
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print("Transformer model trained.")


if __name__ == "__main__":
    train()
