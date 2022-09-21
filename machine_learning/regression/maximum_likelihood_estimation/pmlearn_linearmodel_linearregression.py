# %%
from sklearn.model_selection import train_test_split
from pmlearn.linear_model import LinearRegression
import pmlearn
from IPython.core.interactiveshell import InteractiveShell
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
import pandas as pd
import numpy as np
import os
from warnings import filterwarnings

np.random.seed(12345)
rc = {'xtick.labelsize': 20, 'ytick.labelsize': 20, 'axes.labelsize': 20, 'font.size': 20,
      'legend.fontsize': 12.0, 'axes.titlesize': 10, "figure.figsize": [12, 6]}
sns.set(rc=rc)
InteractiveShell.ast_node_interactivity = "all"

print('Running on pymc-learn v{}'.format(pmlearn.__version__))

# %% Data
X = np.random.randn(1000, 1)
noise = 2 * np.random.randn(1000, 1)
slope = 4
intercept = 3
y = slope * X + intercept + noise
y = np.squeeze(y)

fig, ax = plt.subplots()
ax.scatter(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# %% Model
model = LinearRegression()

# %% Inference (Train)
model.fit(X_train, y_train, inference_type='nuts')

# %% Convergence
pm.traceplot(model.trace, lines={"betas": slope,
                                 "alpha": intercept,
                                 "s": 2},
             varnames=["betas", "alpha", "s"])
pm.gelman_rubin(model.trace, varnames=["betas", "alpha", "s"])
pm.energyplot(model.trace)
pm.forestplot(model.trace, varnames=["betas", "alpha", "s"])

# %% Critize
pm.summary(model.trace, varnames=["betas", "alpha", "s"])
pm.plot_posterior(model.trace, varnames=["betas", "alpha", "s"],
                  figsize=[14, 8])

# collect the results into a pandas dataframe to display
# "mp" stands for marginal posterior
pd.DataFrame({"Parameter": ["betas", "alpha", "s"],
              "Parameter-Learned (Mean Value)": [float(model.trace["betas"].mean(axis=0)),
                                                 float(
                                                     model.trace["alpha"].mean(axis=0)),
                                                 float(model.trace["s"].mean(axis=0))],
              "True value": [slope, intercept, 2]})

# %% Predict
y_predict = model.predict(X_test)
model.score(X_test, y_test)

max_x = max(X_test)
min_x = min(X_test)

slope_learned = model.summary['mean']['betas__0_0']
intercept_learned = model.summary['mean']['alpha__0']
fig1, ax1 = plt.subplots()
ax1.scatter(X_test, y_test)
ax1.plot([min_x, max_x], [slope_learned*min_x + intercept_learned,
                          slope_learned*max_x + intercept_learned], 'r', label='MCMC')
ax1.legend()

# %%
model.save('pickle_jar/linear_model')

# %% Use already trained model for prediction
model_new = LinearRegression()
model_new.load('pickle_jar/linear_model')
model_new.score(X_test, y_test)

#%% Multiple predictors
num_pred = 2
X = np.random.randn(1000, num_pred)
noise = 2 * np.random.randn(1000,)
y = X.dot(np.array([4, 5])) + 3 + noise
y = np.squeeze(y)

model_big = LinearRegression()

model_big.fit(X, y)

model_big.summary