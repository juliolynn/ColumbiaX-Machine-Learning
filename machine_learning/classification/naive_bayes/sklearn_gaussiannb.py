# %% Imports
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import plotly
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode

# %% Preparation
n_samples = 50000
n_bins = 3  # use 3 bins for calibration_curve as we have 3 clusters here

# Generate 3 blobs with 2 classes where the second blob contains
# half positive samples and half negative samples. Probability in this
# blob is therefore 0.5.
centers = [(-5, -5), (0, 0), (5, 5)]
X, y = make_blobs(n_samples=n_samples, centers=centers, shuffle=False,
                  random_state=42)

y[:n_samples // 2] = 0
y[n_samples // 2:] = 1
np.random.seed(42)
sample_weight = np.random.rand(y.shape[0])

# split train, test for calibration
X_train, X_test, y_train, y_test, sw_train, sw_test = \
    train_test_split(X, y, sample_weight, test_size=0.9, random_state=42)

# %% Gaussian Naive-Bayes with no calibration
clf = GaussianNB()
clf.fit(X_train, y_train)  # GaussianNB itself does not support sample-weights
prob_pos_clf = clf.predict_proba(X_test)[:, 1]

# %%
h = .02  # step size in the mesh

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
# Put the result into a color plot
Z = Z.reshape(xx.shape)
Z

# %%
trace = go.Contour(y=xx[0], z=Z, x=xx[0],
                   line=dict(width=0),
                   contours=dict(coloring='heatmap'),
                   opacity=0.7, showscale=False)

# %% Plot the training points
traceArr = []
y_unique = np.unique(y)

for this_y in y_unique:
    this_X = X_train[y_train == this_y]
    this_sw = sw_train[y_train == this_y]
    traceArr.append(go.Scatter(x=this_X, y=this_y, showlegend=False, mode='markers', marker=dict(color=color)))

traceArr.append(trace)

# %% Plot
plotly.offline.iplot({
    "data": traceArr
})
