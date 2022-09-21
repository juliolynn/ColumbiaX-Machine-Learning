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


p = [3.4, 2.8, -10]  # init values for b and c
x = np.linspace(0, 6, 20)
y = f(x, *p) + 20*np.random.normal(size=len(x))  # noise
xn = np.linspace(0, 6, 200)

# %% Train


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
X = addPolynomialOrder(X, 3)
X_Transpose = np.transpose(X)
X_Transpose_X = np.dot(X_Transpose, X)
lambda_Identity = np.identity(X_Transpose_X.shape[0]) * 50
A=lambda_Identity+X_Transpose_X

X_Transpose_y = np.dot(X_Transpose, y)

wRR = np.linalg.solve(A, X_Transpose_y)

# The coefficients
print('Coefficients: \n', wRR)

# %% Plot
# y_new = x_new_Transpose * wLS
Xn = xn.reshape(-1, 1)  # build a matrix of new x vectors
Xn_T = np.transpose(addPolynomialOrder(Xn, 3))
yn = np.dot(wRR, Xn_T)  # reverse the order of factors because of dimesionality

trace0 = go.Scatter(x=x, y=y, mode='markers', name='training data')
trace1 = go.Scatter(x=xn, y=yn, name='predictions')

plotly.offline.iplot({
    "data": [trace0, trace1]
})
