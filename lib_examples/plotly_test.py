#%%
import plotly 
plotly.tools.set_credentials_file(username='jclynn', api_key='1VxgnpuwvLpOJSHmtkaa')

#%%
# Standard plotly imports
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode

# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

#%%
trace0 = go.Scatter(
    x=[1, 2, 3, 4],
    y=[10, 15, 13, 17]
)
trace1 = go.Scatter(
    x=[1, 2, 3, 4],
    y=[16, 5, 11, 9]
)
data = [trace0, trace1]

py.plot(data, filename = 'basic-line', auto_open=True)

#%%
import numpy as np
import pandas as pd

x = np.linspace(0, 20, 100)  # Create a list of evenly-spaced numbers over the range
df = pd.DataFrame(np.sin(x))
print(df)
df.iplot()

#%%
from scipy.stats import norm


gaus = norm.pdf(x, np.mean(x), np.std(x))  
df = pd.DataFrame(gaus)
df.iplot()