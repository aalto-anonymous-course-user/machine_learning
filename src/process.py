import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

console_column = 'console'
title_column = 'title'
labelColumn = 'esrb_rating'

Ratings = {
    'E': 0,
    'ET': 1,
    'T': 2,
    'M': 3
}


def readData(csvName):
    df = pd.read_csv(csvName)
    df.drop(columns=[title_column, console_column], inplace=True)
    df[labelColumn] = df[labelColumn].map(Ratings)
    return df


def getFeatures(data):
    features_count = data.shape[1] - 1
    features = data.drop(columns=[labelColumn],
                         inplace=False).to_numpy().reshape(-1, features_count)
    return features


def getLabels(data):
    return data[labelColumn].to_numpy()
