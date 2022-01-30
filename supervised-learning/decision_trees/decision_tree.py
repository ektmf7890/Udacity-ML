from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

data = np.loadtxt('data.csv', delimiter=',')
X = data[:, :-1]
y = data[:, -1]

model = DecisionTreeClassifier(max_depth=7, min_samples_leaf=1)
model.fit(X, y)

y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)

print("Accuracy: %.2f" % acc)

# plot the decision boundary
# https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html
import matplotlib.pyplot as plt
