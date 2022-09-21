# %%
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import r2_score


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
    return c + b * x + a * x ** 2
    # return b**x+c #(exponential)


p = [3, -2, -10]  # init values for b and c
x = np.linspace(0, 6, 6)
y = f(x, *p) + 20*np.random.normal(size=len(x))  # noise
xn = np.linspace(0, 6, 200)
yn = f(xn, *p) + 20*np.random.normal(size=len(xn))  # noise

#%%
x[:, np.newaxis]

# %% Train


model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                  ('lasso', LassoCV(alphas=[10, 30, 35, 20, 100], cv=5, fit_intercept=False))])
model = model.fit(x[:, np.newaxis], y)

# The coefficients
print('Coefficients: \n', model.named_steps['lasso'].coef_)
print('alphas: \n', model.named_steps['lasso'].alpha_)

# %% Predict
Xn = xn.reshape(-1, 1)
ypred = model.predict(Xn)

# %% Plot
trace0 = go.Scatter(x=x, y=y, mode='markers', name='training data')
trace1 = go.Scatter(x=xn, y=ypred, name='predictions')

plotly.offline.iplot({
    "data": [trace0, trace1]
})

# %%
r2_score_lasso = r2_score(yn, ypred)
print(model)
print("r^2 on test data : %f" % r2_score_lasso)