# %%
import numpy as np
from sklearn import linear_model


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


p = [3.7, 10]  # init values for b and c
x = np.linspace(0, 6, 20)
y = f(x, *p) + 20*np.random.normal(size=len(x))  # noise
xn = np.linspace(0, 6, 200)

# %% Train
reg = linear_model.LinearRegression()
X = x.reshape(-1, 1)
reg.fit(X, y)

# The coefficients
print('Coefficients: \n', reg.coef_)

# %% Plot
Xn = xn.reshape(-1, 1)
yn = reg.predict(Xn)

trace0 = go.Scatter(x=x, y=y, mode='markers')
trace1 = go.Scatter(x=xn, y=yn)

plotly.offline.iplot({
    "data": [trace0, trace1],
    "layout": go.Layout(title="hello world")
})


# %%


# %%
