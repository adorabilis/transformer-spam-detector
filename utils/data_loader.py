from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    data_path = Path(__file__).parent.parent / "data" / "sms_spam_collection.csv"

    df = pd.read_csv(
        data_path,
        names=["label", "text"],
        usecols=[0, 1],
        skiprows=1,
        encoding="latin1",
    )
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["text"] = df["text"].str.lower().str.replace(r"[^\w\s]", "", regex=True)

    # print(df.head(10))
    return train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
