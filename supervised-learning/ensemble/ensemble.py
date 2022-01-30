from matplotlib.colors import ListedColormap
from sklearn import ensemble
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ensemble_compare():
    iris = load_iris()
    
    X = iris.data[:, [0, 1]] # sepal length, sepal width
    y = iris.target

    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    ada_boost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=30)
    random_forest = RandomForestClassifier(n_estimators=30)
    bagging = BaggingClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=30)
    gradient_boosting = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=3)

    # plot range
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_range = np.linspace(x_min, x_max, 100)
    y_range = np.linspace(y_min, y_max, 100) 
    xx, yy = np.meshgrid(x_range, y_range)

    for i, clf in enumerate((gradient_boosting, ada_boost, random_forest, bagging)):
        clf.fit(X, y)

        plt.subplot(2, 2, i + 1)
        
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(x_range, y_range, Z, cmap=plt.cm.RdYlBu, alpha=0.7)
        
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['r', 'y', 'b']), edgecolors='k', s=20)

    plt.show()

if __name__ == "__main__":
    ensemble_compare()