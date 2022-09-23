from plotting import generate_confusion_matrix, generate_labels_histrogram, generate_rating_and_feature_scatter_plot
from process import getFeatures, getLabels, readData
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

esrb_csv = './rawdata/Video_games_esrb_rating.csv'

fig, axes = plt.subplots(3, 1)
fig.tight_layout(pad=2)


data = readData(esrb_csv)
features = getFeatures(data)
labels = getLabels(data)


generate_labels_histrogram(axes[0], labels, 'Distribution of ratings')
generate_rating_and_feature_scatter_plot(axes[1], features, labels)

# Split dataset into training and validation set
X_train, X_val, y_train, y_val = train_test_split(
    features, labels, test_size=0.33, random_state=12)

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

generate_confusion_matrix(axes[2], y_val, y_pred)

plt.show()

print('accuracy of LogisticRegression', accuracy)
