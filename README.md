# DistilBERT Spam Detector

This project is a learning exercise focused on text classification using the DistilBERT transformer-based model. The goal is to explore the capabilities of natural language processing (NLP) in identifying spam messages. I also wanted to experiment with the Hugging Face Transformers library.

- I fined-tuned the DistilBERT base model on the SMS Spam Collection dataset which consists of spam and non-spam messages to create a binary classification model.
- `Transformers` library was used to load the pre-trained model, tokenise the data, and train the model. Different hyperparameters were tested to optimise performance.
- For comparison, I also implemented a logistic regression model to evaluate the performance of a more traditional machine learning approach.

## Instructions

```shell
# Train the logistic regression model
python train_lr.py

# Train the transformer model
python train_transformer.py

# Evaluate the models
python evaluate.py
```

Sample output:

```
Traditional Model Metrics:
Precision: 0.855
Recall: 0.907
F1-Score: 0.880

Transformer Model Metrics:
Precision: 0.980
Recall: 0.973
F1-Score: 0.977
```

| ![Confusion_Matrices](https://github.com/user-attachments/assets/d055290e-8ed5-4ea2-acd7-c0a2ea0cbde0) | ![Model_Comparison](https://github.com/user-attachments/assets/5ec44270-cbc0-46ec-bba6-615f877314cd) |
|---|---|
