# %%
import numpy as np
from scipy import optimize
import pandas as pd

# Standard plotly imports
import plotly
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode

# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

# %% Data


def f(x, a, b):
    return a*np.sin(b*np.pi*x)


p = [5, 5]
x = np.linspace(0, 1, 30)
y = f(x, *p) + .5*np.random.normal(size=len(x))
xn = np.linspace(0, 1, 200)

# %% Train


def residual(p, x, y):
    return y - f(x, *p)


p0 = [3, 4]
popt, pcov = optimize.curve_fit(f, x, y, p0=p0)
print(popt)

yn = f(xn, *popt)

# %% Plot
yn = f(xn, *popt)

trace0 = go.Scatter(x=x, y=y, mode='markers')
trace1 = go.Scatter(x=xn, y=yn)

plotly.offline.iplot({
    "data": [trace0, trace1],
    "layout": go.Layout(title="hello world")
})
