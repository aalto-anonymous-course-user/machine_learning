from collections import Counter

import matplotlib.axes as axes
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # data visualization library
from sklearn.metrics import confusion_matrix
import pandas as pd

fontsize = 10


def generate_training_and_validation_plot(ax: axes.Axes, x, training_accuracies, validation_accuracies):
    ax.plot(x, training_accuracies, label='Training accuracy')
    ax.plot(x, validation_accuracies, label='Validation accuracy')
    ax.set_title(
        'Training and Validation Accuracy vs Max depth of Tree', fontsize=fontsize)
    ax.set_xlabel("Max Depth of Tree", fontsize=fontsize)
    ax.set_ylabel('Accuracy', fontsize=fontsize)
    ax.legend()


def generate_feature_histogram(ax: axes.Axes, data: pd.DataFrame, features: np.ndarray, labels: np.ndarray, title: str):
    # visualize the confusion matrix
    columns = data.drop(columns=['esrb_rating'], inplace=False).columns
    df = pd.DataFrame(columns=['feature', 'rating'])
    features_size = features.shape[0]

    for i in range(features_size):
        for c in range(columns.size):
            row = features[i]
            feature_name = columns[c]
            rating = labels[i]
            has_feature = row[c] == 1
            if has_feature:
                df.loc[len(df.index)] = [feature_name, rating]

    sorted = df.sort_values(by=['feature'])
    sns.countplot(data=sorted, ax=ax, x="feature", hue="rating")
    ax.set_xlabel('Features', fontsize=fontsize)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(),
                  rotation=45, ha='right')
    ax.set_ylabel('Count', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)


def generate_labels_histogram(ax: axes.Axes, labels: np.ndarray, title: str):
    ax.hist(labels)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("ESRB Rating", fontsize=fontsize)
    ax.set_ylabel('Number of Data Points', fontsize=fontsize)


# Copied and modified from Course Assignment 2
def generate_confusion_matrix(ax: axes.Axes, y_true: np.ndarray, y_pred: np.ndarray, title: str):
    # visualize the confusion matrix
    c_mat = confusion_matrix(y_true, y_pred)
    sns.heatmap(c_mat, annot=True, fmt='g', ax=ax,
                xticklabels=['1', '2', '3', '4'], yticklabels=['1', '2', '3', '4'])
    ax.set_xlabel('Predicted ratings', fontsize=fontsize)
    ax.set_ylabel('True ratings', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)


def generate_rating_and_feature_scatter_plot(ax: axes.Axes, features: np.ndarray, labels: np.ndarray, title: str):
    features_size = features.shape[0]
    feature_counts = np.zeros(features_size)
    for i in range(features_size):
       # print(i)
        row = features[i]
        feature_counts[i] = np.sum(row)

    # count the occurrences of each point
    c = Counter(zip(feature_counts, labels))
    # create a list of the sizes, here multiplied by 10 for scale
    s = [1.2*c[(xx, yy)]
         for xx, yy in zip(feature_counts, labels)]

    ax.scatter(feature_counts,
               labels, s=s)
    ax.set_xlabel("Content Descriptor Count", fontsize=fontsize)
    ax.set_ylabel("Rating", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
