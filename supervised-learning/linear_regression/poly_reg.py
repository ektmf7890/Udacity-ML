import numpy as np

train_data = np.loadtxt('poly_data.csv', delimiter=',', skiprows=1)

X = train_data[:, :-1]
y = train_data[:, -1]
# print(X.shape, y.shape)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

degree = 4

poly_feat = PolynomialFeatures(degree)
X_poly = poly_feat.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# make pipeline
poly_pipeline = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly_pipeline.fit(X, y)

import matplotlib.pyplot as plt
X_seq = np.linspace(X.min(), X.max(), 300).reshape(-1,1)

plt.figure()
plt.scatter(X, y)
plt.plot(X_seq, poly_model.predict(poly_feat.fit_transform(X_seq)), color='black')
plt.show()