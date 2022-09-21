# Source: https://anujkatiyal.com/blog/2017/09/30/ml-regression/

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


p = [2.8, 1.3, -10]  # init values for b and c
x = np.linspace(0, 6, 20)
y = f(x, *p) + 20*np.random.normal(size=len(x))  # noise
xn = np.linspace(0, 6, 200)

# %% Train


def SolveRidgeRegression(X, y, lam):
    # (lambda * Identity + X_Transpose * X)_Inverse * X_Transpose * y
    X_Transpose = np.transpose(X)
    X_Transpose_X = np.dot(X_Transpose, X)
    lambda_Identity = np.identity(X_Transpose_X.shape[0]) * lam
    XTX_Inverse = np.linalg.inv(lambda_Identity + X_Transpose_X)
    X_Transpose_y = np.dot(X_Transpose, y)
    wRR = np.dot(XTX_Inverse, X_Transpose_y)
    return wRR


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


X = x.reshape(-1, 1)

wRR = SolveRidgeRegression(addPolynomialOrder(X, 3), y, 10000)

# The coefficients
print('Coefficients: \n', wRR)

# %% Plot
# y_new = x_new_Transpose * wLS
Xn = xn.reshape(-1, 1)  # build a matrix of new x vectors
Xn_T = np.transpose(addPolynomialOrder(Xn, 3))
yn = np.dot(wRR, Xn_T)  # reverse the order of factors because of dimesionality

trace0 = go.Scatter(x=x, y=y, mode='markers')
trace1 = go.Scatter(x=xn, y=yn)

plotly.offline.iplot({
    "data": [trace0, trace1],
    "layout": go.Layout(title="Ridge Regression from scratch (using numpy)")
})
