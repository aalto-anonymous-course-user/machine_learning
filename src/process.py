from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

console_column = 'console'
title_column = 'title'
label_column = 'esrb_rating'
no_descriptors_column = 'no_descriptors'

games_missing_no_descriptors = [
    "Pop-Up Pilgrims",
    "Old Man's Journey",
    "Cooking Mama: Cookstar",
    "Harvest Moon: Mad Dash",
    "The Angry Birds Movie 2 VR: Under Pressure",
    "Everybody's Golf VR",
    "Old Man's Journey",
    "Garfield Kart: Furious Racing",
    "Race with Ryan",
    "Project Highrise - Architect's Edition",
    "The Golf Club 2019 Featuring the PGA Tour",
    "Race with Ryan Road Trip Deluxe Edition",
    "Transport Giant",
    "Infinite Minigolf",
    "Conga Master",
    "The Golf Club 2",
    "Gem Smashers",
    "Gran Turismo Sport",
    "Parappa the Rapper Remastered"
]

Ratings = {
    'E': 1,
    'ET': 2,
    'T': 3,
    'M': 4
}


def read_data(csvName: str):
    df = pd.read_csv(csvName)
    # Some games have no_descriptors set incorrectly
    for game_with_missing_no_descriptors in games_missing_no_descriptors:
        df.loc[df[title_column] == game_with_missing_no_descriptors,
               no_descriptors_column] = 1

    df.drop(columns=[title_column, console_column], inplace=True)
    df[label_column] = df[label_column].map(Ratings)
    return df


def get_features_dataframe(data: pd.DataFrame, drop_no_descriptors: bool):
    features_count = data.shape[1] - 1  # -1 as we drop esrb_rating
    columns_to_drop = [label_column]
    if (drop_no_descriptors):
        columns_to_drop.append(no_descriptors_column)
        features_count -= 1

    features = data.drop(columns=columns_to_drop,
                         inplace=False)
    return features


def get_features(data: pd.DataFrame, drop_no_descriptors: bool):
    features_df = get_features_dataframe(data, drop_no_descriptors)
    features_count = data.shape[1] - 1  # -1 as we drop esrb_rating
    return features_df.to_numpy().reshape(-1, features_count)


def get_labels(data):
    return data[label_column].to_numpy()
