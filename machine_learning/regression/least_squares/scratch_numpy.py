# %%
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
    return c + b * x + x ** a
    # return b**x+c #(exponential)


p = [3, 2, -10]  # init values for b and c
x = np.linspace(0, 6, 20)
y = f(x, *p) + 20*np.random.normal(size=len(x))  # noise
xn = np.linspace(0, 6, 200)

# %% Train


def SolveLeastSquares(X, y):
    # (X_Transpose * X)_Inverse * X_Transpose * y
    X_Transpose = np.transpose(X)
    X_Transpose_X = np.dot(X_Transpose, X)
    XTX_Inverse = np.linalg.inv(X_Transpose_X)
    X_Transpose_y = np.dot(X_Transpose, y)
    wLS = np.dot(XTX_Inverse, X_Transpose_y)
    return wLS


X = x.reshape(-1, 1)
wLS = SolveLeastSquares(X, y)

# The coefficients
print('Coefficients: \n', wLS)

# %% Plot
# y_new = x_new_Transpose * wLS
Xn = xn.reshape(-1, 1) # build a matrix of new x vectors
Xn_T = np.transpose(Xn)
yn = np.dot(wLS, Xn_T) # reverse the order of factors because of dimesionality

trace0 = go.Scatter(x=x, y=y, mode='markers')
trace1 = go.Scatter(x=xn, y=yn)

plotly.offline.iplot({
    "data": [trace0, trace1],
    "layout": go.Layout(title="Least Squares from scratch (using numpy)")
})
