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

# TODO Write in functional way
# TODO Name Graphs and axes
# TODO Insert into streamlit logic

def data_importer():
   
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
   
   return stock_data

def rolling_stats(stock_data):

   #Determine rolling statistics
   rollmean = stock_data.rolling(12).mean()
   rollstd = stock_data.rolling(12).std()

   plt.figure(figsize=(16,7))
   fig = plt.figure(1)

   #Plot rolling statistics:
   orig = plt.plot(stock_data, color='blue',label='Original')
   mean = plt.plot(rollmean, color='red', label='Rolling Mean')
   std = plt.plot(rollstd, color='black', label = 'Rolling Std')
   plt.legend(loc='best')
   plt.title('Rolling Mean & Standard Deviation')
   plt.show()

def stationary(stock_data):

   #Making the series stationary
   plt.figure(figsize=(16,7))
   fig = plt.figure(1)


   ts_log = np.log(stock_data)
   plt.title('Log Value of Closing Number')
   plt.plot(ts_log)
   plt.show()
   
   return ts_log

def decomposition(ts_log):
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

def difference(ts_log):
   #Differencing of Time Series
   plt.figure(figsize=(16,7))
   fig = plt.figure(1)
   ts_log_diff = ts_log - ts_log.shift()
   plt.plot(ts_log_diff)
   
   return ts_log_diff


# TODO Redundant? rolling_stats()
# #Determining rolling statistics
# rolLmean = ts_log_diff.rolling(12).mean()
# rolLstd = ts_log_diff.rolling(12).std()


# #Plot rolling statistics:
# orig = plt.plot(ts_log_diff, color='blue',label='Original')
# mean = plt.plot(rolLmean, color='red', label='Rolling Mean')
# std = plt.plot(rolLstd, color='black', label = 'Rolling Std')
# plt.legend(loc='best')
# plt.title('Rolling Mean & Standard Deviation')
# plt.show(block=False)

def check_autocorrelation(stock_data, ts_log_diff):
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

def ARIMA_Values(ts_log_diff, ts_log, stock_data):
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
   
   return results_ARIMA

def ARIMA_prediction(results_ARIMA):
   print(results_ARIMA.predict(10,20))
   
def main():
   # Import stock data
   stock_data = data_importer()
   # Get rolling stats for ARIMA
   rolling_stats(stock_data)
   # Make data stationary(stock_data)
   ts_log = stationary(stock_data)
   # Decompose data?
   decomposition(ts_log)
   # What is the reason for ts log diff? Change over time?
   ts_log_diff = difference(ts_log)
   # Check to which degree data is autocorrelated
   check_autocorrelation(stock_data, ts_log_diff)
   # Train ARIMA model and make prediction
   results_ARIMA = ARIMA_Values(ts_log_diff, ts_log, stock_data)
   # Pring ARIMA prediction
   ARIMA_prediction(results_ARIMA)
   
   # Can this model make future predictions or just be tested on test data when trained?
