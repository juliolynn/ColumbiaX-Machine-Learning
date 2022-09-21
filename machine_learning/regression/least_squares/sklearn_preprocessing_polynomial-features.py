# %%
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np


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
    return c + b * x  + x ** a
    # return b**x+c #(exponential)


p = [3, 2, -10]  # init values for b and c
x = np.linspace(0, 6, 20)
y = f(x, *p) + 20*np.random.normal(size=len(x))  # noise
xn = np.linspace(0, 6, 200)

# %% Train


model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                  ('linear', LinearRegression(fit_intercept=False))])
model = model.fit(x[:, np.newaxis], y)

# The coefficients
print('Coefficients: \n', model.named_steps['linear'].coef_)

# %% Plot
Xn = xn.reshape(-1, 1)
yn = model.predict(Xn)

trace0 = go.Scatter(x=x, y=y, mode='markers')
trace1 = go.Scatter(x=xn, y=yn)

plotly.offline.iplot({
    "data": [trace0, trace1],
    "layout": go.Layout(title="Scikit Learn Polynomial Regression")
})
