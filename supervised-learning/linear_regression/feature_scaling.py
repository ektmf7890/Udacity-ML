import numpy as np

train_data = np.loadtxt('reg_data.csv', delimiter=',')
X = train_data[:, :-1]
y = train_data[:, -1]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.linear_model import LinearRegression
lin_model = LinearRegression()
lin_model.fit(X, y)
print('Linear Regression: ', lin_model.coef_)

from sklearn.linear_model import Lasso
lasso_reg = Lasso()
lasso_reg.fit(X, y)
reg_coef = lasso_reg.coef_
print('No feature scaling', reg_coef)

scaled_lasso_reg = Lasso()
scaled_lasso_reg.fit(X_scaled, y)
print('Feature scaling: ', scaled_lasso_reg.coef_)