"""
Train the model
"""
import numpy as np
import torch
from typing import Iterable
import pandas as pd
from model import train_model, embed, WADataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import pickle

def compute_accuracy(
        embedding_model,
        model,
        data: pd.DataFrame,
        target = "age",
        predictors: Iterable = ("cue", "R1Raw", "R2Raw", "R3Raw"),
        device = "cpu",
        only_relevant = False
):
    """
    Compute the acciracy of a model on
    a given target
    """
    if target == "age":
        data["age"] = data["age"].apply(lambda age: 0 if age < 30 else 1 if age < 50 else 2 if age < 70 else 3)
        print(f"age value densities: {data['age'].value_counts(normalize=True)}")

    if only_relevant:
        data = data[data["cue"].isin(("drug", "name", "first name", "country", "disease", "illness", "color", "colour"))]
        print(f"relevant dataset size: {len(data)}")

    labels = data[target]
    predictors = data[list(predictors)]

    dataset = WADataset(predictors, labels)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    for X, y in dataloader:
        pass

    X_embedding = embed(embedding_model, X)
    y_logits = model(X_embedding)
    y_pred = torch.argmax(y_logits, dim=1)
    n_correct = torch.sum(y_pred == y).item()

    return confusion_matrix(y, y_pred), n_correct / len(y)

if __name__ == "__main__":

    df = pd.read_csv("preprocessing/SWOW-EN.complete_preprocessed.csv")
    print(len(df))

    # filter out nan
    df = df[df["cue"] != np.nan]

    df_shuffled = df.dropna().sample(frac=1)
    df_train = df_shuffled[:int(len(df_shuffled) * 0.8)]
    df_eval = df_shuffled[int(len(df_shuffled) * 0.8):]

    embedding_model, model = train_model(df_train, n_epochs=3)

    print("EVALUATION ON ALL DATA")

    conf_mat, accuracy = compute_accuracy(embedding_model, model, df_eval)

    print(f"evaluation accuracy: {accuracy}")

    print("confusion matrix:")
    print(conf_mat)

    print("ONLY RELEVANT DATA")

    conf_mat, accuracy = compute_accuracy(embedding_model, model, df_eval, only_relevant=True)

    print(f"evaluation accuracy: {accuracy}")

    print("confusion matrix:")
    print(conf_mat)

    with open("data/trained_model.p", "wb") as fp:
        pickle.dump(model, fp)
