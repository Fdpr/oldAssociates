"""
This file contains the model we use to predict demographic variables from word association data.
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Iterable
import fasttext
from tqdm import tqdm

EMBEDDINGS_PATH = "data/wiki-news-300d-1M-subword.vec"

def get_fasttext_embedding(word: str, embedding_model: fasttext.FastText):
    """
    Returns a fasttext embedding for a given word, returning zero for non-responses
    """
    if type(word) != str or word == "No more responses":
        return np.zeros(300, dtype=np.float32)
    else:
        return embedding_model.get_word_vector(word)


class WADataset(Dataset):
    """
    A word association dataset.
    The predictors consist of cues and three responses,
    the labels are the demographic variable we are trying to predict.
    """

    def __init__(self, predictors: pd.DataFrame, labels: pd.DataFrame):

        # the dataset consists of lists of predictors and labels
        # TODO: allow for more (or less) than three responses
        self.predictors = list(predictors.apply(lambda row: [row["cue"], [row["R1Raw"], row["R2Raw"], row["R3Raw"]]], axis=1))
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.predictors[idx], self.labels[idx]


class WordAssociationPredictionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(600, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, 4)

    def forward(self, x: torch.Tensor):
        """
        Takes in a tokenized cue and response set and returns a prediction
        """
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.activation(self.linear4(x))
        x = self.linear5(x)
        return x



def embed(embedding_model, predictors: list):
    """
    Returns a cue and response embedding given
    """
    cues, all_responses = predictors
    cue_vectors = torch.tensor(np.array([get_fasttext_embedding(cue, embedding_model) for cue in cues]))
    response_vectors = torch.tensor(np.array([[get_fasttext_embedding(response, embedding_model) for response in responses] for responses in all_responses]))
    # TODO: do the geometrically-weighted average in a more extensible way
    response_vectors = torch.mean(response_vectors.transpose(0,2) * torch.tensor([1, 0.5, 0.25]), dim=-1).t()

    return torch.cat([cue_vectors, response_vectors], dim=1)

def train_model(
        data: pd.DataFrame,
        target: str = "age",
        predictors: Iterable = ("cue", "R1Raw", "R2Raw", "R3Raw"),
        device: str = "cpu",
        n_epochs = 10
    ):
    """
    Train the model on a given dataframe
    """
    embedding_model = fasttext.load_model('data/crawl-300d-2M-subword.bin')
    model = WordAssociationPredictionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # TODO: convert data to categories in preprocessing
    if target == "age":
        data["age"] = data["age"].apply(lambda age: 0 if age < 30 else 1 if age < 50 else 2 if age < 70 else 3)

    # shape the data into desired format
    labels = data[target]
    predictors = data[list(predictors)]

    dataset = WADataset(predictors, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(n_epochs):
        print(f"epoch {epoch}")
        running_loss = 0
        # train the model
        for X, y in tqdm(dataloader):

            optimizer.zero_grad()
            X_embedding = embed(embedding_model, X)
            y_logits = model(X_embedding)

            loss = loss_fn(y_logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"mean loss: {running_loss / len(dataloader)}")

    return embedding_model, model
