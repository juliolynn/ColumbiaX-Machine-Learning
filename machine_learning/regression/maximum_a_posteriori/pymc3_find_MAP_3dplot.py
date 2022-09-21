# %%
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

# Standard plotly imports
import plotly
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode

# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

# %% Data
# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
axes[0].scatter(X1, Y)
axes[1].scatter(X2, Y)
axes[0].set_ylabel('Y')
axes[0].set_xlabel('X1')
axes[1].set_xlabel('X2')


# %% Train
basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=2)
    sigma = pm.HalfNormal('sigma', sd=1)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

# %%
map_estimate = pm.find_MAP(model=basic_model)

map_estimate

# %%

with basic_model:
    # draw 500 posterior samples
    trace = pm.sample()

# %%
_ = pm.traceplot(trace)
pm.summary(trace)


# %%
alphan = trace.get_values('alpha')[0]
betan = trace.get_values('beta')[0, :]

# %% Plot
X1n = np.random.randn(size)
X2n = np.random.randn(size) * 0.2

Yn = alphan + betan[0]*X1n + betan[1]*X2n  # + np.random.randn(size)*sigma

# %%
trace0 = go.Scatter3d(x=X1, y=X2, z=Y, mode='markers', name='training')
trace1 = go.Mesh3d(x=X1n, y=X2n, z=Yn,  name='predictions')

plotly.offline.iplot({
    "data": [trace0, trace1]
})
