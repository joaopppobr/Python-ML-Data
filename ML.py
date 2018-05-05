import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

##For the pretty graphs
style.use('ggplot')

##Import the data from quandl and assign it to dataframes
df = quandl.get('WIKI/GOOGL')

##Limit the information
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]

##Create a porcentage and a porcentage change just for more clear info
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())

#Create the forecasts
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True)

forecast_out = int (math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out:]

df.dropna(inplace = True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

## Here I just comment out the part where the algorithm is trained
"""""
clf = LinearRegression(n_jobs = -1)
clf.fit(X_train, y_train)

with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)
"""

#Import the trained algorithm, so I dont have to train it again
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)    

##I define accuracy and the forecast set.
accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy)

## A loop that serves just to define the timestamp for the predictions,
## so I can graph the data
df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range (len(df.columns) - 1)] + [i]
    
##Here I plot everything out
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
    
    

