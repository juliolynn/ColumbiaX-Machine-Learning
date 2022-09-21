# %%
import numpy as np
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import pandas as pd
from scipy import stats

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
    return x*b+c
    # return b**x+c #(exponential)


N = 100
p = [3.7, 10]  # init values for b and c
x = np.random.randn(N)
y = f(x, *p) + 2*np.random.normal(size=len(x))  # noise
X = sm.add_constant(x)

xn = np.linspace(x.min(), x.max(), 50)

#df = pd.DataFrame({'y':y, 'x':x})
#df['constant'] = 1

# %% Train
# sm.OLS(df.y,df[['constant','x']]).fit().summary()
model = sm.OLS(y, X)
fitted = model.fit()

# %%
#wML = fitted.params
# wML
# %%
# fitted.tvalues

# %%
#print(fitted.t_test([0, 1]))

# %%
# print(fitted.f_test(np.identity(2)))

# The coefficients
#print('Coefficients: \n', reg.coef_)

# %% Predict
Xn = sm.add_constant(xn)
yn = fitted.predict(Xn)

# %% Confidence
y_hat = fitted.predict(X)  # predict y for original training data
y_err = y - y_hat

mean_x = np.mean(x)

dof = N - fitted.df_model - 1
t = stats.t.ppf(1-0.05, df=dof)
s_err = np.sum(np.power(y_err, 2))  # sum of squared errors

conf = t * np.sqrt((s_err/(N-2))*(1.0/N + (np.power((xn-mean_x), 2) /
                                           ((np.sum(np.power(xn, 2))) - N*(np.power(mean_x, 2))))))

upper = yn + abs(conf)

lower = yn - abs(conf)

# %% Prediction interval
sdev, lower_int, upper_int = wls_prediction_std(fitted, exog=Xn, alpha=0.05)

# %%Plot
data_trace = go.Scatter(x=x, y=y, stackgroup='model', mode='markers')

fitted_trace = go.Scatter(x=xn, y=yn, stackgroup='model')

upper_conf_trace = go.Scatter(
    x=xn, y=upper, stackgroup='conf', fill=None, line=dict(color='rgba(122, 0, 0, 0.2)'))
lower_conf_trace = go.Scatter(x=xn, y=lower, stackgroup='conf',
                              fill='tonexty', line=dict(color='rgba(122, 0, 0, 0.2)'))

upper_int_trace = go.Scatter(x=xn, y=upper_int, stackgroup='int',
                       fill=None, line=dict(color='rgba(50, 50, 50, 0.2)'))
lower_int_trace = go.Scatter(x=xn, y=lower_int, stackgroup='int',
                       fill='tonexty', line=dict(color='rgba(50, 50, 50, 0.2)'))


plotly.offline.iplot({
    "data": [
        data_trace,
        fitted_trace,
        upper_conf_trace,
        lower_conf_trace,
        upper_int_trace, 
        lower_int_trace
    ]
})
