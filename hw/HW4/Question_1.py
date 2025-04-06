#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 12:25:17 2025

@author: aj
"""

import pandas as pd
import numpy as np
import pydotplus
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from io import BytesIO
import PIL.Image
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

SAMPLE_SIZE = 10

# Random Seed
np.random.seed(42)
random.seed(42)


def generate_data():
    # Features are age, BP, BMI
    ages = np.random.randint(30, 90, size=SAMPLE_SIZE)
    bps = np.random.randint(90, 181, size=SAMPLE_SIZE)
    bmis = np.round(np.random.uniform(18.5, 40.0, size=SAMPLE_SIZE), 1)

    # Generate Labels
    labels = []
    for age, bp in zip(ages, bps):
        if age > 60 and bp > 140:
            risk = 1
        else:
            risk = 0

        if random.random() < 0.2:
            risk = 1 - risk
        labels.append(risk)

    # Combine into the final dataset
    data = list(zip(ages, bps, bmis, labels))

    # Return a dataframe
    df = pd.DataFrame(data, columns=['Age', 'BP', 'BMI', 'Stroke'])

    return df


def img_tree(tree, feature_names):
    # Convert tree to DOT Format
    tree_dot = export_graphviz(tree, feature_names, filled=True, out_file=None)

    img_graph = pydotplus.graph_from_dot_data(tree_dot)
    img = PIL.Image.open(BytesIO(img_graph.create_png()))

    # Display the img
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Trains and test the model


def model(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Train-test split (80%train, 20%test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Evaluate the performance
    print(f"Accuracy: {accuracy * 100:.4}%")
    # print("\nClassification Report:\n", classification_report(y_test, y_pred))


stroke_df = generate_data()
model(stroke_df)
