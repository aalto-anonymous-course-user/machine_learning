import pydotplus
from sklearn.tree import export_graphviz, DecisionTreeClassifier
import pandas as pd


def create_tree_pdf(clf: DecisionTreeClassifier, file_name: str, features: pd.DataFrame):
    d_tree = export_graphviz(clf, feature_names=list(
        features.columns), filled=True, class_names=["E", "ET", 'T', 'M'])
    pydot_graph = pydotplus.graph_from_dot_data(d_tree)
    pydot_graph.write_pdf(file_name)
