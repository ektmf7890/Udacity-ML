import numpy as np

train_data = np.loadtxt('reg_data.csv', delimiter=',')
X = train_data[:, :-1]
y = train_data[:, -1]

from sklearn.linear_model import Lasso

lasso_reg = Lasso()
lasso_reg.fit(X, y)

reg_coef = lasso_reg.coef_
print('Lasso: ', reg_coef) # you will see that the first and sixth column has been zeroed out (feature selection)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

lin_coef = lin_reg.coef_
print('Linear Regression: ', lin_coef)