
# https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import plotly
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode

# %%
n_neighbors = 15

# import some data to play with
iris = datasets.load_iris()
iris

# %%
# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target


# %%
h = .02  # step size in the mesh


# preprocess dataset, split into training and test part
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)

# %%
Xy_train = np.append(X_train, y_train.reshape(-1, 1), 1)
X_train_class_0 = np.asarray(Xy_train[Xy_train[:, 2] == 0])[:, :2]
X_train_class_1 = np.asarray(Xy_train[Xy_train[:, 2] == 1])[:, :2]
X_train_class_2 = np.asarray(Xy_train[Xy_train[:, 2] == 2])[:, :2]
Xy_test = np.append(X_test, y_test.reshape(-1, 1), 1)
X_test_class_0 = np.asarray(Xy_test[Xy_test[:, 2] == 0])[:, :2]
X_test_class_1 = np.asarray(Xy_test[Xy_test[:, 2] == 1])[:, :2]
X_test_class_2 = np.asarray(Xy_test[Xy_test[:, 2] == 2])[:, :2]

weights = 'uniform'
# for weights in ['uniform', 'distance']:
# %%
# we create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
clf.fit(X, y)

# %%
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# %%
xx

# %%
yy

# %%
# just plot the dataset first
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# %%
Z


# %%
# Put the result into a color plot
Z = Z.reshape(xx.shape)
Z

# %%
trace = go.Contour(y=yy[0], z=Z, x=xx[0],
                   showscale=False, connectgaps=True, zsmooth='best',
                   opacity=0.7, )

# %%
# Plot the training points
training_points_class_0 = go.Scatter(x=X_train_class_0[:, 0], y=X_train_class_0[:, 1], showlegend=False,
                                     mode='markers', marker=dict(color='blue'))
training_points_class_1 = go.Scatter(x=X_train_class_1[:, 0], y=X_train_class_1[:, 1], showlegend=False,
                                     mode='markers', marker=dict(color='gray'))
training_points_class_2 = go.Scatter(x=X_train_class_2[:, 0], y=X_train_class_2[:, 1], showlegend=False,
                                     mode='markers', marker=dict(color='red'))

# and testing points
test_points_class_0 = go.Scatter(x=X_test_class_0[:, 0], y=X_test_class_0[:, 1], showlegend=False,
                                 mode='markers', marker=dict(color='blue'))
test_points_class_1 = go.Scatter(x=X_test_class_1[:, 0], y=X_test_class_1[:, 1], showlegend=False,
                                 mode='markers', marker=dict(color='gray'))
test_points_class_2 = go.Scatter(x=X_test_class_2[:, 0], y=X_test_class_2[:, 1], showlegend=False,
                                 mode='markers', marker=dict(color='red'))
'''
plotly.offline.iplot({
    "data": [training_points_class_0, training_points_class_1, training_points_class_2, trace]
})

plotly.offline.iplot({
    "data": [test_points_class_0, test_points_class_1, test_points_class_2, trace]
})
'''

# %%
plotly.offline.iplot({
    "data": [
        training_points_class_0, training_points_class_1, training_points_class_2,
        test_points_class_0, test_points_class_1, test_points_class_2,
        trace
    ]
})
