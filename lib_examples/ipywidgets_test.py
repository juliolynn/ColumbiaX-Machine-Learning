#%%
import pandas as pd
from numpy.random import randn

import ipywidgets as widgets
from ipywidgets import interact, interact_manual

dataframe1= pd.DataFrame(randn(4,3),['A','B','C','D',],['X','Y','Z'])

@interact
def show_articles_more_than(column='X', x=2):
    return dataframe1.loc[dataframe1[column] > x]

