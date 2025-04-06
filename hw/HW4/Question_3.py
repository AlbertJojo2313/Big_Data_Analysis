#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 13:30:50 2025

@author: aj
"""

import pandas as pd
import pydotplus
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import PIL.Image
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score  # Import accuracy_score

def img_tree(tree, feature_names):
    # Convert tree to DOT Format
    tree_dot = export_graphviz(tree, feature_names=feature_names, filled=True, out_file=None)

    img_graph = pydotplus.graph_from_dot_data(tree_dot)
    


    # Save the img
    img_graph.write_png('tree_output.png')

    # Display the img
    img = PIL.Image.open('tree_output.png')
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def model():
    data = load_iris(as_frame=True)

    X = data.frame.iloc[:, :-1]
    y = data.frame.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Show the image of the tree
    img_tree(model, feature_names=X.columns)  

    accuracy = accuracy_score(y_test, y_pred)  
    # Evaluate the model
    print(f"Accuracy: {accuracy * 100:.4f}%")

model()
