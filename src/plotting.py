import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns  # data visualization library
from collections import Counter


def generate_labels_histrogram(ax, labels, title: str):
    ax.hist(labels)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_title(title)
    ax.set_xlabel("ESRB Rating")
    ax.set_ylabel('Number of Data Points')


# Copied and modified from Course Assignment 2
def generate_confusion_matrix(ax, y_true: np.ndarray, y_pred: np.ndarray):
    # visualize the confusion matrix
    c_mat = confusion_matrix(y_true, y_pred)
    sns.heatmap(c_mat, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted labels', fontsize=15)
    ax.set_ylabel('True labels', fontsize=15)
    ax.set_title('Confusion Matrix', fontsize=15)


def generate_rating_and_feature_scatter_plot(ax, features: np.ndarray, labels: np.ndarray):
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
    ax.set_xlabel("Content Descriptor Count")
    ax.set_ylabel("Rating")
    ax.set_title("Content Descriptor Count vs Rating ")
