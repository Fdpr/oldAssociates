"""
Train the model
"""
import numpy as np
import torch
from typing import Iterable
import pandas as pd
from model import train_model, WADataset
from torch.utils.data import DataLoader

def compute_accuracy(
        model,
        data: pd.DataFrame,
        target = "age",
        predictors: Iterable = ("cue", "R1Raw", "R2Raw", "R3Raw"),
        device = "cpu"
):
    """
    Compute the acciracy of a model on
    a given target
    """
    if target == "age":
        data["age"] = data["age"].apply(lambda age: 0 if age < 30 else 1 if age < 60 else 2)
        print(f"age value densities: {data['age'].value_counts(normalize=True)}")

    labels = data[target]
    predictors = data[list(predictors)]

    dataset = WADataset(predictors, labels)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    for X, y in dataloader:
        pass

    X_embedding = model.embed(X)
    y_logits = model(X_embedding)
    y_pred = torch.argmax(y_logits, dim=1)
    n_correct = torch.sum(y_pred == y).item()

    return n_correct / len(dataset)

if __name__ == "__main__":

    df = pd.read_csv("preprocessing/SWOW-EN.complete_preprocessed.csv")

    # filter out nan
    df = df[df["cue"] != np.nan]

    df_shuffled = df.dropna().sample(frac=1)
    df_train = df_shuffled[:int(len(df_shuffled) * 0.8)]
    df_eval = df_shuffled[int(len(df_shuffled) * 0.8):]

    model = train_model(df_train, n_epochs=1)

    accuracy = compute_accuracy(model, df_eval)

    print(f"evaluation accuracy: {accuracy}")
