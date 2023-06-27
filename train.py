"""
Train the model
"""
import pandas as pd
from model import train_model

if __name__ == "__main__":

    df = pd.read_csv("preprocessing/SWOW-EN.complete_preprocessed.csv")

    model = train_model(df)

    print(model)
