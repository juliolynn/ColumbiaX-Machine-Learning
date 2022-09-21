#%%
import pandas as pd
from numpy.random import randn

dataframe1= pd.DataFrame(randn(4,3),['A','B','C','D',],['X','Y','Z'])

print(dataframe1)

#%%
import matplotlib.pyplot as plt
import numpy as np

x = dataframe1["X"]
y = dataframe1["Y"]

plt.plot(x, y)      
plt.show()  