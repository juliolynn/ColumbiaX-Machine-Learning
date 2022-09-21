# %%
import numpy as np
from scipy import optimize

# Standard plotly imports
import plotly
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode

# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

# %% Data


def f(x, b, c):
    return x**b+c
    # return b**x+c #(exponential)


p = [1.8, 10]  # init values for b and c
x = np.linspace(0, 6, 20)
y = f(x, *p) + np.random.normal(size=len(x))  # noise
xn = np.linspace(0, 6, 200)

# %% Train


def residual(p, x, y):
    return y - f(x, *p)


p0 = [1., 8.]
tr_options = dict(damp=500)
wRR = optimize.least_squares(residual, p0, tr_solver='lsmr', tr_options=tr_options, args=(x, y))

# %% Plot
yn = f(xn, *wRR.x)

trace0 = go.Scatter(x=x, y=y, mode='markers')
trace1 = go.Scatter(x=xn, y=yn)

plotly.offline.iplot({
    "data": [trace0, trace1],
    "layout": go.Layout(title="hello world")
})
