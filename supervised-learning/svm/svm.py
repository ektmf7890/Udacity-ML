from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def svc_plot():
    data = np.asarray(pd.read_csv("data.csv", header=None))
    X = data[:, :-1]
    y = data[:, -1]

    x_range = np.linspace(0.05, 1.05, 500)
    y_range = np.linspace(0.05, 1.05, 500)
    xx, yy = np.meshgrid(x_range, y_range)

    C = 1.0
    svc = SVC(kernel='linear', C=C).fit(X, y)
    poly_svc = SVC(kernel='poly', C=10.0, degree=3).fit(X, y)
    rbf_svc = SVC(kernel='rbf', C=C, gamma=30).fit(X, y)

    titles = ['SVC with linear kernel',
            'SVC with RBF kernel',
            'SVC with polynomial (degree 3) kernel']

    for i, clf in enumerate((svc, poly_svc, rbf_svc)):
        plt.subplot(2, 2, i + 1)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        
        # print(Z.shape)
        # print(np.c_[xx.ravel(), yy.ravel()])

        Z = Z.reshape(xx.shape)
        plt.contourf(x_range, y_range, Z, cmap=plt.cm.coolwarm, alpha=0.7)

        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
        plt.xlim(xx.min()-0.05, xx.max()+0.05)
        plt.ylim(yy.min()-0.05, yy.max()+0.05)
        plt.xticks(())
        plt.yticks(())

    plt.show()

if __name__ == "__main__":
    svc_plot()
    # data = np.asarray(pd.read_csv("data.csv", header=None))
    # plt.hist(data[:, 1], bins=10)
    # plt.show()