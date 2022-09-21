# %%
from scipy.stats import norm
import numpy as np
import pandas as pd

# Standard plotly imports
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode

# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

# %%
x = np.linspace(0, 6, 20)
y = x**2 + 2 * np.random.normal(size=len(x))
xn = np.linspace(0, 6, 200)
print(y)

# %%
popt = np.polyfit(x, y, 3)
yn = np.polyval(popt, xn)

trace0 = go.Scatter(x=x, y=y, mode='markers')
trace1 = go.Scatter(x=xn, y=yn)

plotly.offline.iplot({
    "data": [trace0, trace1],
    "layout": go.Layout(title="hello world")
})
