from unittest import result
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.create_tree_pdf import create_tree_pdf

from src.plotting import (generate_confusion_matrix, generate_feature_histogram, generate_labels_histogram,
                          generate_rating_and_feature_scatter_plot, generate_training_and_validation_plot)
from src.process import get_features_dataframe, get_features, get_labels, read_data

test_csv = './rawdata/test_esrb.csv'
esrb_csv = './rawdata/Video_games_esrb_rating.csv'


data = read_data(esrb_csv)
features_df = get_features_dataframe(data, False)
features = get_features(data, False)
labels = get_labels(data)

test_data = read_data(test_csv)
test_features = get_features(test_data, False)
test_labels = get_labels(test_data)

# Split dataset into training and validation set
X_train, X_val, y_train, y_val = train_test_split(
    features, labels, test_size=0.33, random_state=12)


def plot_dataset():
    fig_a, axes_a = plt.subplots(1, 2)
    fig_a.tight_layout(pad=3)
    generate_labels_histogram(
        axes_a[0], labels, 'Distribution of ratings (training & validation)')
    generate_labels_histogram(
        axes_a[1], test_labels, 'Distribution of ratings (test)')
    plt.show()
    fig_b, axes_b = plt.subplots(1, 2)
    fig_b.tight_layout(pad=3)

    features_without_no_descriptor = get_features(data, True)
    test_features_without_no_descriptor = get_features(test_data, True)
    generate_rating_and_feature_scatter_plot(
        axes_b[0], features_without_no_descriptor, labels, "Content Descriptor Count vs Rating (training & validation)")
    generate_rating_and_feature_scatter_plot(
        axes_b[1], test_features_without_no_descriptor, test_labels, "Content Descriptor Count vs Rating (test)")
    plt.show()
    fig_c, axes_c = plt.subplots(2, 1)
    fig_c.tight_layout(pad=3)
    generate_feature_histogram(
        axes_c[0], data, features, labels, 'Features and Rating Count (training & validation)')
    generate_feature_histogram(
        axes_c[1], test_data, test_features, test_labels, 'Features and Rating Count (test)')
    plt.show()


def get_logistic_regression_model():
    fig, axes = plt.subplots(2, 1)
    fig.tight_layout(pad=3)

    log_clf = LogisticRegression()
    log_clf.fit(X_train, y_train)
    y_pred_train = log_clf.predict(X_train)
    y_pred_val = log_clf.predict(X_val)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    generate_confusion_matrix(
        axes[0], y_train, y_pred_train, 'Confusion matrix of training data (Logistic Regression)')
    generate_confusion_matrix(
        axes[1], y_val, y_pred_val, 'Confusion matrix of validation data (Logistic Regression)')
    plt.show()

    print('Training accuracy of Logistic Regression', accuracy_train)
    print('Validation accuracy of Logistic Regression', accuracy_val)
    return log_clf


def get_decision_trees_model():
    fig, axes = plt.subplots(3, 1)
    fig.tight_layout(pad=3)
    best_clf = DecisionTreeClassifier(
        criterion='gini', random_state=55)
    best_clf.fit(X_train, y_train)
    y_pred_train = best_clf.predict(X_train)
    y_pred_val = best_clf.predict(X_val)
    best_validation_accuracy = accuracy_score(y_val, y_pred_val)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    print('No depth restriction training accuracy of Decision Trees', accuracy_train)
    print('No depth restriction validation accuracy of Decision Trees',
          best_validation_accuracy)
    print('------')

    depth_labels = range(32)
    accuracy_train_values = [accuracy_train]
    accuracy_val_values = [best_validation_accuracy]

    best_depth = 'None'

    for depth in range(1, 32):
        clf = DecisionTreeClassifier(
            criterion='gini', max_depth=depth, random_state=55)
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_val = clf.predict(X_val)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_val = accuracy_score(y_val, y_pred_val)
        accuracy_train_values.append(accuracy_train)
        accuracy_val_values.append(accuracy_val)

        print('For max_depth:', depth)
        print('Training accuracy of Decision Trees:', accuracy_train)
        print('Validation accuracy of Decision Trees:', accuracy_val)
        if accuracy_val > best_validation_accuracy:
            best_depth = depth
            best_validation_accuracy = accuracy_val
            best_clf = clf
    y_pred_train = best_clf.predict(X_train)
    y_pred_val = best_clf.predict(X_val)
    generate_confusion_matrix(
        axes[0], y_train, y_pred_train, 'Decision Tree Confusion matrix of training data with max_depth ' + str(best_depth))
    generate_confusion_matrix(
        axes[1], y_val, y_pred_val, 'Decision Tree Confusion matrix of validation data with max_depth ' + str(best_depth))
    generate_training_and_validation_plot(
        axes[2], depth_labels, accuracy_train_values, accuracy_val_values)
    plt.show()
    best_accuracy_train = accuracy_score(y_train, y_pred_train)
    best_accuracy_val = accuracy_score(y_val, y_pred_val)
    print('Best training accuracy of Decision Trees', best_accuracy_train)
    print('Best validation accuracy of Decision Trees', best_accuracy_val)
    pdf_title = "best_decision_tree_max_depth_{:s}.pdf".format(str(best_depth))
    create_tree_pdf(best_clf, pdf_title, features_df)
    return best_clf


def compare_logistic_and_decision_trees():
    logistic_clf = get_logistic_regression_model()
    decison_tree_clf = get_decision_trees_model()
    logistic_pred = logistic_clf.predict(test_features)
    decision_tree_pred = decison_tree_clf.predict(test_features)

    logistic_accuracy = accuracy_score(test_labels, logistic_pred)
    decision_tree_accuracy = accuracy_score(test_labels, decision_tree_pred)

    fig, axes = plt.subplots(2, 1)
    fig.tight_layout(pad=3)

    generate_confusion_matrix(
        axes[0], test_labels, logistic_pred, 'Logistic Regression Confusion matrix with test dataset')
    generate_confusion_matrix(
        axes[1], test_labels, decision_tree_pred, 'Decision Tree Confusion matrix with test dataset')

    plt.show()

    print('Logistic Regression Test Set accuracy', logistic_accuracy)
    print('Decision Tree accuracy', decision_tree_accuracy)
