# https://blog.applied.ai/bayesian-inference-with-pymc3-part-1/

# %%
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import pandas as pd

# Standard plotly imports
import plotly
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode

# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

# %%


def addPolynomialOrder(input_matrix, p):
    if p == 1:
        return input_matrix
    elif p == 2:
        a = input_matrix
        b = np.power(input_matrix[:, 0:6], 2)
        out = np.hstack((a, b))
        return out
    elif p == 3:
        a = input_matrix
        b = np.power(input_matrix[:, 0:6], 2)
        c = np.power(input_matrix[:, 0:6], 3)
        out = np.hstack((a, b, c))
        return out


# %% Data
# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha, sigma = 10, 25
beta = [-3.5, 12.8, 40.7]

# Size of dataset
size = 100

# Predictor variable
x = np.random.randn(size)
X = x.reshape(-1, 1)

# Simulate outcome variable
y = alpha + beta[0]*x**3 + beta[1]*x**2 + beta[2]*x + np.random.randn(size)*sigma


# %% Train
basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    # alpha = pm.Normal('alpha', mu=0, sd=10)
    # beta = pm.Normal('beta', mu=0, sd=10, shape=3)
    # sigma = pm.HalfNormal('sigma', sd=1)

    X_Poly = addPolynomialOrder(X, 3)

    # Expected value of outcome

   # mu = alpha + beta[0]*X_Poly[:, 0]+beta[1] * \
    #    X_Poly[:, 1]+beta[2]*X_Poly[:, 2]

    # Likelihood (sampling distribution) of observations
    # Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

    pm.glm.GLM.from_formula('y ~ 1 + x', dict(x=X_Poly, y=y),
                            family=pm.glm.families.Normal())


# %%
#start_MAP = pm.find_MAP(model=basic_model)

# start_MAP
# TODO:
# find_MAP should not be used to initialize the NUTS sampler,
# simply call pymc3.sample() and it will automatically initialize NUTS in a better way.

# %%

with basic_model:
    # take samples using NUTS
    # trace = pm.sample(2000, start=start_MAP, step=pm.NUTS())
    trace = pm.sample(2000)

# %%
_ = pm.traceplot(trace)
pm.summary(trace)

# %%
df_trace = pm.trace_to_dataframe(trace)

# %% extract traces
trc_mu = df_trace[['Intercept', 'x[0]', 'x[1]', 'x[2]']]
trc_sd = df_trace['sd']

# %% recreate the likelihood
N = 1000
xn = np.linspace(x.min(), x.max(), N)
Xn = xn.reshape(-1, 1)
Xn_Poly = np.append(np.ones(N).reshape(-1, 1),
                    addPolynomialOrder(Xn, 3),
                    1)

# %%
like_mu = np.dot(Xn_Poly, trc_mu.T)  # mu_n = xn_Transposed * mu
# sigma_n_sq = sigma_sq + xn_Transposed * SIGMA * xn
like_sd = np.tile(trc_sd.T, (N, 1))
like = np.random.normal(like_mu, like_sd)

# %% Credible Regions
dfp = pd.DataFrame(np.percentile(like, [2.5, 25, 50, 75, 97.5], axis=1).T, columns=[
                   '025', '250', '500', '750', '975'])
dfp['x'] = xn

# %% Plot
data_trace = go.Scatter(x=x, y=y, mode='markers')
fitted_trace = go.Scatter(x=dfp['x'], y=dfp['500'])

# Confidence or Credible Regions
upper_conf_trace = go.Scatter(x=dfp['x'], y=dfp['750'], stackgroup='conf',
                              fill=None, line=dict(color='rgba(122, 0, 0, 0.2)'))
lower_conf_trace = go.Scatter(x=dfp['x'], y=dfp['250'], stackgroup='conf',
                              fill='tonexty', line=dict(color='rgba(122, 0, 0, 0.2)'))

upper_int = go.Scatter(x=dfp['x'], y=dfp['975'], stackgroup='int',
                       fill=None, line=dict(color='rgba(50, 50, 50, 0.2)'))
lower_int = go.Scatter(x=dfp['x'], y=dfp['025'], stackgroup='int',
                       fill='tonexty', line=dict(color='rgba(50, 50, 50, 0.2)'))

plotly.offline.iplot({
    "data": [
        data_trace,
        fitted_trace,
        upper_conf_trace,
        lower_conf_trace,
        upper_int, lower_int
    ]
})
