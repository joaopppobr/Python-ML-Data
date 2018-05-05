import pandas as pd
from pandas import DataFrame
import datetime

from pandas_datareader import data
import pandas_datareader.data as web
import matplotlib.pyplot as plt

df = pd.read_csv('Data/airline_safety.csv')

print (df.head())

df[['incidents_00_14', 'fatal_accidents_00_14']].plot()
plt.show()