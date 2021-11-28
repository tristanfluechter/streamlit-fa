"""
This module uses the stock dataframe to create and train a
LSTM model to predict stock prices at a certain date.

****************************************************************
Credit for the overall usage of an LSTM model for predicting stock prices: https://towardsdatascience.com/time-series-forecasting-with-recurrent-neural-networks-74674e289816
All customizing and re-writing as functional code has been done by the authors of this assignment.
****************************************************************
"""

# Import relevant libraries
from matplotlib.pyplot import close
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
import plotly.graph_objects as go
import streamlit as st


def lstm_prepare_data(stockdata, stockticker):
    """
    This program prepares the data for a LSTM-model by getting a user input on the train-test-split.
    To avoid overfitting, data should always be split into training and testing data. After training a model with one LSTM layer,
    the program visualizes the given predictions. Those predictions are then used to give an estimate of where the stock could be in the next n days.
    
    Returns: look_back, date_train, date_test, close_train, close_test, train_generator, test_generator, close_data_noarray, close_data
    """
    # To get a sensible train test split, we restrict the amount of freedom our user gets.
    split_percent = st.number_input("Please enter your desired train-test-split as a positive float between 0.6 and 0.8: ", min_value=0.6, max_value=0.8, value=0.8)
    
    # Convert dataframe index (which is a datetime index) to a column
    stockdata['Date'] = stockdata.index

    # Create a closing price list (will be used later!)
    close_data_noarray = stockdata['Close'].values
    
    # Create a closing price array
    close_data = close_data_noarray.reshape((-1,1))
    
    # Split the data into train and test data
    split = int(split_percent*len(close_data))
    close_train = close_data[:split]
    close_test = close_data[split:]

    date_train = stockdata['Date'][:split]
    date_test = stockdata['Date'][split:]
    
    # Consider past 10 days
    look_back = 10

    # Generate a dataset to train the LSTM. Each datapoint has the shape of ([past 15 values], [current value]).
    train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)     
    test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)
    
    return look_back, date_train, date_test, close_train, close_test, train_generator, test_generator, close_data_noarray, close_data
    
    
def lstm_train(look_back, train_generator, test_generator, close_test, close_train):
    """
    This model trains an LSMT model based on the data provided by
    lstm_prepare_data().

    Returns: model, prediction, close_train, close_test
    """
    # Create the LSTM model
    model = Sequential()
    
    # Add first (and only) layer to the model. 15 nodes, ReLu activation function 
    model.add(LSTM(units = 15, activation='relu', input_shape=(look_back,1)))
    
    # Create densely connected NN layer
    model.add(Dense(1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mse')
    
    # How many epochs do we train the model?
    num_epochs = 50
    
    # Fit model with data, don't allow status updates
    model.fit(train_generator, epochs=num_epochs, verbose=1)
    
    # Create prediction values
    prediction = model.predict(test_generator)

    # Reshape train, test and prediction data to plot them
    close_train = close_train.reshape((-1)) 
    close_test = close_test.reshape((-1)) 
    prediction = prediction.reshape((-1))
    
    return model, prediction, close_train, close_test

def lstm_visualize(date_test, date_train, close_test, close_train, prediction, stockticker):
    # Create plotly line 1: Training data
    trace1 = go.Scatter(
        x = date_train,
        y = close_train,
        mode = 'lines',
        name = 'Data'
    )
    # Create plotly line 2: Testing data
    trace2 = go.Scatter(
        x = date_test,
        y = prediction,
        mode = 'lines',
        name = 'Prediction'
    )
    # Create plotly line 3: Prediction Values
    trace3 = go.Scatter(
        x = date_test,
        y = close_test,
        mode='lines',
        name = 'Ground Truth'
    )
    
    # Show Figure
    fig = go.Figure(data=[trace1, trace2, trace3])
    
    # Format layout (Show legend in top left corner,
    # remove title margins, remove backgroud colour,
    # label the axes
    fig.layout.update(showlegend = True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), 
                      margin=go.layout.Margin(l=60, r=0, b=0, t=30),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(title="Date"),yaxis=dict(title="Closing Price in USD"))
    
    # Format axes: Border
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    
    # Plot train / test performance and use full column width
    st.plotly_chart(fig, use_container_width = True)

def lstm_make_prediction(model, look_back, stockdata, close_data, close_data_noarray, stockticker):
    """
    Makes a prediction for the next 30 days by appending forecast data to prediction
    dataframe using the LSTM model. Returns last day of prediction.
    """
    
    # For how many days do we predict?
    # Set to 15 because the LSTM predictions autocorrelate with previous predictions.
    # This way, we avoid overly dramatic movements in either direction.
    num_prediction = 15
    
    prediction_list = close_data[-look_back:] # input scaled data
    
    for number in range(num_prediction):
        
        x = prediction_list[-look_back:] # create list with last look_back values
        x = x.reshape((1, look_back, 1)) # reshape last values so it fits the data
        out = model.predict(x)[0][0] # create a prediction based on those values
        prediction_list = np.append(prediction_list, out) # append this prediction to the output
        
    prediction_list = prediction_list[look_back-1:]
    # Get last date of dataframe
    last_date = stockdata['Date'].values[-1]
    # Create prediction dates time series
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()

    # Plot original values and dates together with predicted values and dates
    trace_original = go.Scatter(
        x = stockdata['Date'][-50:],
        y = close_data_noarray[-50:],
        mode = 'lines',
        name = 'Original Price Curve'
    )

    trace_pred = go.Scatter(
        x = prediction_dates,
        y = prediction_list,
        mode = 'lines',
        name = 'Prediction'
    )

    # Show Figure
    fig2 = go.Figure(data=[trace_original, trace_pred])
    
    # Add title
    fig2.layout.update(showlegend = True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), 
                      margin=go.layout.Margin(l=60, r=0, b=0, t=30),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(title="Date"),yaxis=dict(title="Closing Price in USD"))
    
    fig2.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig2.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    
    st.plotly_chart(fig2, use_container_width=True)
    
    lstm_pred = int(prediction_list[-1])
    
    return lstm_pred

def lstm_evaluation(prediction, close_train):
    root_mean_square_error = np.sqrt(((prediction[0] - close_train[0]) ** 2).mean()).round(2)
    st.write(f"The trained LSTM model shows an RSME of {root_mean_square_error}.")