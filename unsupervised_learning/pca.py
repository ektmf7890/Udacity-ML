#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('./mnist_train.csv')
train.shape


# In[3]:


train.head()


# In[4]:


y = train['label']
X = train.drop('label', axis=1)


# In[5]:


sns.countplot(y, color=sns.color_palette()[0])


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[7]:


def fit_random_forest_cls(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestClassifier()

    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print('Acc: %.4f' %acc)

    # plot confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(cf_matrix, annot=True, cmap='bwr')

fit_random_forest_cls(X, y)


# In[8]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[9]:


def do_pca(n_componenets, data):
    X = StandardScaler().fit_transform(data)

    pca = PCA(n_components=n_componenets)
    X_pca = pca.fit_transform(data)
    return pca, X_pca


# In[10]:


pca, X_pca = do_pca(2, X)


# In[11]:


X_pca.shape


# In[12]:


pca.explained_variance_ratio_


# In[13]:


fit_random_forest_cls(X_pca, y)


# In[14]:


pca.components_.shape


# In[21]:


np.min(X_pca, axis=0)


# In[25]:


X_pca[:100].shape


# In[28]:


def plot_components(X, y):
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)

    n_samples = X.shape[0]
    for i in range(n_samples):
        plt.text(X[i][0], X[i][1], str(y[i]), color=plt.cm.Set1(y[i]), fontdict={'size':15})

    plt.xticks([])
    plt.yticks([])
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)

plot_components(X_pca[:100], y[:100])


# In[43]:


def show_images(n_images):
    for i in range(n_images):
        plt.subplot(n_images//10, 10, i + 1)
        plt.imshow(X.iloc[i].values.reshape(28, 28), cmap='binary')
        plt.xticks([])
        plt.yticks([])

show_images(100)


# In[54]:


def show_images_by_digit(digit):
    images = X[y == digit].values[:50]
    for i in range(50):
        plt.subplot(5, 10, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='binary')
        plt.yticks([])
        plt.xticks([])
show_images_by_digit(4)


# In[ ]:




