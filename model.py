"""
This file contains the model we use to predict demographic variables from word association data.
"""
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from typing import Iterable
import fasttext

EMBEDDINGS_PATH = "data/wiki-news-300d-1M-subword.vec"

class WordAssociationPredictionModel(nn.Module):

    def __init__(self, embedding_model):
        super().__init__()

        self.activation = nn.ReLU()
        self.embedding_model = embedding_model
        self.linear1 = nn.Linear(600, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 3)

    def forward(self, x):
        """
        Takes in a tokenized cue and response set and returns a prediction
        """
        embedding = self.embed(x["cue"], x["responses"])
        x = self.activation(self.linear1(embedding))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x


    def embed(self, cue: str, responses: Iterable):
        """
        Returns a cue and response embedding given
        """

        cue_vector = self.embedding_model.get_word_vector(cue)
        all_response_vectors = torch.tensor([self.embedding_model.get_word_vector(response) for response in responses])
        response_vector = torch.mean(all_response_vectors.dot(torch.tensor([1, 0.5, 0.25])), dim=0)

        return cue_vector.concat(response_vector)

def train_model(
        data: pd.DataFrame,
        target: str = "age",
        predictors: Iterable = (""),
        device: str = "cpu",
        n_epochs = 10
    ):
    """
    Train the model on a given dataframe
    """
    embedding_model = fasttext.load_model("cc.en.300.bin")
    model = WordAssociationPredictionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # shape the data into desired format
    y_train = torch.tensor(data[target], device=device)
    X_train = torch.tensor(data[predictors], device=device)

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(n_epochs):
        # train the model
        for X, y in dataloader:
            optimizer.zero_grad()
            y_logits = model(X)

            loss = loss_fn(y_logits, y)
            loss.backward()
            optimizer.step()

    return model
