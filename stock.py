import quandl
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn import datasets,linear_model
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
import matplotlib.pyplot as plt

df=quandl.get("WIKI/AMZN")
print(df.tail())
df=df[['Close']]
forecast_out=int(10)
df['Prediction']=df[['Close']].shift(-forecast_out)
print(df.head(20))
#define features Matrix X by excluding the label column which we just created.
X = np.array(df.drop(['Prediction'],1))
#using a feature in sklearn preprocessing to scale features.
X = preprocessing.scale(X)
#X contains last 'n=forecast_out' rows for which we dont have label data.
X_forecast = X[-forecast_out:]
X = X[:-forecast_out]
#similarly define label vector y for the data we have prediction for.

y=np.array(df['Prediction'])
y=y[:-forecast_out]
#cross validation(split into test and train data).
#test_size=0.2==>20% data ios test data.
X_train,X_test,y_train,y_test =cross_validation.train_test_split(X,y,test_size=0.2)
#train.
clf=LinearRegression()
clf.fit(X_train,y_train)
#test.
accuracy=clf.score(X_test,y_test)
print("Accuracy:",accuracy)
#predict using our model.
forecast_prediction=clf.predict(X_forecast)
print(forecast_prediction)
df.dropna(inplace=True)
df['forecast']=np.nan
last_date=df.iloc[-1].name
last_unix=last_date.timestamp()
one_day =86400
next_unix=last_unix+one_day
for i in forecast_prediction:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix+=86400
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]
    df['Close'].plot(figsize=(12,8),color="green")
    df['forecast'].plot(figsize=(12,8),color="red")
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
#split in a year.
df['Close'].plot(figsize=(10,5),color="blue")
df['forecast'].plot(figsize=(10,5),color="yellow")
plt.xlim(xmin=datetime.date(2016,2,1))
plt.ylim(ymin=500)
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Close')
plt.show()
