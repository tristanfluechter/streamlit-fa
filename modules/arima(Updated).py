import warnings
warnings.filterwarnings('ignore')
# Data and package Import
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA



#Read the data from Yahoo and import
while True:
   try:
            stock_ticker = str(input("Please enter the stock ticker of the stock you want to analyze: "))
            start_date = str(input("Please enter a start date for stock analysis (YYYY-DD-MM): "))
            end_date = str(input("Please enter an end date for stock analysis (YYYY-MM-DD): "))
            # Create dataframe with DataReader module
            stock_data = data.DataReader(stock_ticker, "yahoo", start_date, end_date)
            break
        
   except:
            print("Invalid format for either stock ticker or dates - please try again and ensure correct format.")
    
# stock_ticker = 'cat'
# start_date = '2020-01-01'
# end_date = '2020-12-01'
TempData = data.DataReader(stock_ticker, "yahoo", start_date, end_date)
TempData.head(30)

#Data Cleaning
stock_data = TempData.dropna()

stock_data = stock_data["Close"][start_date:end_date]
print(stock_data.describe())

# Explore the stock performance over the given time period. 
plt.figure(figsize=(16,7))
fig = plt.figure(1)
ax1 = fig.add_subplot(111)
ax1.set_xlabel('Time Frame')
ax1.set_ylabel('Stock Price for CAT')
ax1.plot(stock_data)
plt.show()


#Determing rolling statistics
rolLmean = stock_data.rolling(12).mean()
rolLstd = stock_data.rolling(12).std()

plt.figure(figsize=(16,7))
fig = plt.figure(1)

#Plot rolling statistics:
orig = plt.plot(stock_data, color='blue',label='Original')
mean = plt.plot(rolLmean, color='red', label='Rolling Mean')
std = plt.plot(rolLstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show()


#Making the series stationary

plt.figure(figsize=(16,7))
fig = plt.figure(1)


ts_log = np.log(stock_data)
plt.title('Log Value of Closing Number')
plt.plot(ts_log)
plt.show()


#Decomposition

decomposition = seasonal_decompose(ts_log,freq=1,model = 'multiplicative')

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(16,7))
fig = plt.figure(1)

plt.title('Decomposition of Time Series')
plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
#plt.show()


#Differencing of Time Series
plt.figure(figsize=(16,7))
fig = plt.figure(1)
ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)

#Determining rolling statistics
rolLmean = ts_log_diff.rolling(12).mean()
rolLstd = ts_log_diff.rolling(12).std()


#Plot rolling statistics:
orig = plt.plot(ts_log_diff, color='blue',label='Original')
mean = plt.plot(rolLmean, color='red', label='Rolling Mean')
std = plt.plot(rolLstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)

#Checking Auto-Corelation
stock_data.sort_index(inplace= True)

lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20)


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(ts_log_diff.dropna(),lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(ts_log_diff.dropna(),lags=40,ax=ax2)
plt.show()

# Add ARIMA Values to the plot



ts_log_diff = ts_log_diff[~ts_log_diff.isnull()] #ts_log_diff.dropna()

plt.figure(figsize=(16,8)) #ts_log_diff.dropna(inplace=True)
model = ARIMA(ts_log_diff, order=(2,1,2))  
results_ARIMA = model.fit()  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('Fitting ARIMA values to the Data')
plt.show()

# Take results back to original scale and predicting stock prices
ARIMA_diff_predictions = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(ARIMA_diff_predictions.head())

ARIMA_diff_predictions_cumsum = ARIMA_diff_predictions.cumsum()
print(ARIMA_diff_predictions_cumsum.head())

ARIMA_log_prediction = pd.Series(ts_log.iloc[0], index=ts_log.index)
ARIMA_log_prediction = ARIMA_log_prediction.add(ARIMA_diff_predictions_cumsum,fill_value=0)
ARIMA_log_prediction.head()

plt.figure(figsize=(12,8))
predictions_ARIMA = np.exp(ARIMA_log_prediction)
plt.plot(stock_data, color = 'blue', label='Actual Value')
plt.plot(predictions_ARIMA, color = 'orange', label = 'Predicted Value')
plt.legend(loc='best')
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-stock_data)**2)/len(stock_data)))
plt.show()

print(results_ARIMA.predict(10,20))
