# Source: https://anujkatiyal.com/blog/2017/09/30/ml-regression/

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


def f(x, a, b, c):
    return c + b * x + x ** a
    # return b**x+c #(exponential)


p = [2.8, 1.3, -10]  # init values for b and c
x = np.linspace(0, 6, 20)
y = f(x, *p) + np.random.normal(size=len(x), scale=5)  # noise
xn = np.linspace(0, 6, 200)

# %% Train
sd = np.std(x)
X = x.reshape(-1, 1)
X_Transpose = np.transpose(X)
X_Transpose_X = np.dot(X_Transpose, X)

XTX_Inverse = np.linalg.inv(X_Transpose_X)
XTy = np.dot(X_Transpose, y)


sigma = sd**(-2) * XTX_Inverse
mu = np.dot(XTX_Inverse, XTy)
print(sigma)
print(mu)

data = dict(x=x, y=y)

# %%


def glm_mcmc_inference(x, y, iterations=5000):
    """
    Calculates the Markov Chain Monte Carlo trace of
    a Generalised Linear Model Bayesian linear regression 
    model on supplied data.

    df: DataFrame containing the data
    iterations: Number of iterations to carry out MCMC for
    """
    # Use PyMC3 to construct a model context
    basic_model = pm.Model()
    with basic_model:
        # Create the glm using the Patsy model syntax
        # We use a Normal distribution for the likelihood
        pm.glm.GLM.from_formula('y ~ x', data, family=pm.glm.families.Normal())

        # Use Maximum A Posteriori (MAP) optimisation as initial value for MCMC
        start = pm.find_MAP()

        # Use the No-U-Turn Sampler
        step = pm.NUTS()

        # Calculate the trace
        trace = pm.sample(
            iterations, step, start,
            random_seed=42, progressbar=True
        )

    return trace


trace = glm_mcmc_inference(x, y, iterations=5000)
pm.traceplot(trace[500:])
plt.show()


# %% Plot
print(trace)
pm.plot_posterior_predictive_glm(trace,samples=100)
'''
yn = 1

trace0 = go.Scatter(x=x, y=y, mode='markers', name='training data')
trace1 = go.Scatter(x=xn, y=yn, name='predictions')

plotly.offline.iplot({
    "data": [trace0, trace1]
})
'''
