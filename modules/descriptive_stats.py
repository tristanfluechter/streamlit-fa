"""
This module uses a created Pandas Dataframe to conduct descriptive statistics.
"""
# Import relevant libraries
from yahoo_fin import stock_info as si # to import stock data
from pandas.plotting import autocorrelation_plot # to get autocorrelation plot
import pandas_ta as ta # to calculate MACD
import matplotlib.pyplot as plt # to display autocorrelation plot
import matplotlib.dates as mdates # to convert dates to numeric
import numpy as np # for fast calculations
import plotly.graph_objects as go # to display graphs
from plotly.subplots import make_subplots # to create subplots
import streamlit as st # to guarantee streamlit functionality

def plot_stockdata(stockdata):
    """
    Plots stock data for a given stock.
    """
    
    # Create plotly object        
    fig = go.Figure()
    
    # Plot closing prices
    fig.add_trace(go.Scatter(x=stockdata.index, y=stockdata.Close, name = "Closing Price"))
    
    # Format layout (Show legend in top left corner, show rangeslider, 
    # remove title margins, remove backgroud colour,
    # label the axes
    fig.layout.update(showlegend = True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), 
                      xaxis_rangeslider_visible = True, margin=go.layout.Margin(l=60, r=0, b=0, t=30),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(title="Date"),yaxis=dict(title="Closing Price in USD"))
    
    # Show chart axes lines
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    
    #Show the graph
    st.plotly_chart(fig, use_container_width = True)

def show_stock_price(ticker, delta, trading_volume):
    """
    A program that returns the current stock price.
    """
    # Get current stock price
    current_stock_price = float(si.get_live_price(ticker).round(2))
    # Display current stock price as metric with change
    st.subheader(f"Current stock price of {ticker}: ")
    st.metric("USD", current_stock_price, delta)
    st.write(f"Trading volume: {trading_volume}")

    return current_stock_price


def describe_stock_data(stockdata, ticker):
    """
    A program that describes the stock data, 
    providing basic descriptive statistics.
    """
    # Save new dataframe for descriptive statistics
    descriptive_df = stockdata.describe().Close
    
    # Get descriptive variables through indexing
    stock_des_mean = descriptive_df['mean'].round(2)
    stock_des_quart1 = descriptive_df['25%'].round(2)
    stock_des_quart2 = descriptive_df['50%'].round(2)
    stock_des_quart3 = descriptive_df['75%'].round(2)
    stock_des_stddev = descriptive_df['std'].round(2)
    stock_des_range = (stock_des_quart3 - stock_des_quart1).round(2)
    stock_des_var_coefficient = ((stock_des_stddev / stock_des_mean) * 100).round(2)
    
    # Create dictionary with descriptive values
    descriptive_data = {
        "Mean closing price":stock_des_mean,
        "First quartile of stock prices":stock_des_quart1,
        "Second quartile of stock prices":stock_des_quart2,
        "Third quartile of stock prices":stock_des_quart3,
        "Quartile range of stock prices":stock_des_range,
        "Standard deviation":stock_des_stddev,
        "Variation coefficient":stock_des_var_coefficient   
    }
    
    return descriptive_data

def plot_trendline(stockdata):
    """
    A program that plots the ticker data over the given timeframe
    and provides a linear trendline.
    """
    # Create plotly object        
    fig = go.Figure()
    
    # Plot closing prices
    fig.add_trace(go.Scatter(x=stockdata.index, y=stockdata.Close, name = "Closing Price"))
    
    # Convert Date Axis to numerical for trend line
    numeric_dates = mdates.date2num(stockdata.index)
    
    # Fit data to create trend line
    fitted_data = np.polyfit(numeric_dates, stockdata.Close, 1)
    trend_curve = np.poly1d(fitted_data)
    
    # Plot trend line
    fig.add_trace(go.Scatter(x=stockdata.index, y=trend_curve(numeric_dates), line=dict(dash = "dash"), name="Trend Line"))
    
    # Format layout (Show legend in top left corner, 
    # remove title margins, remove backgroud colour,
    # label the axes
    fig.layout.update(showlegend = True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), 
                      margin=go.layout.Margin(l=60, r=0, b=0, t=30),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(title="Date"),yaxis=dict(title="Closing Price in USD"))
    
    # Show chart axes lines
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        
    #Show the graph
    st.plotly_chart(fig)


def plot_simple_ma(stockdata):
    """
    A program that plots the ticker data over the given timeframe
    and provides moving averages based on user input.
    """
    # Get total number of days in dataframe
    max_ma_value = int(len(stockdata) * 0.7)
    
    # Define moving averages to plot
    # Error handling
    try:    
        ma1_input = st.number_input("Please state a first moving average to plot (in days): ", min_value=1, max_value = max_ma_value, value=10, step=1)
        ma2_input = st.number_input("Please state a second moving average to plot (in days): ", min_value=1, max_value = max_ma_value, value=20, step=1)
            
    except:
        st.write("Invalid input. Please enter a positive integer.")
    
    # Create moving averages based on user input
    ma1 = stockdata.Close.rolling(ma1_input).mean()
    ma2 = stockdata.Close.rolling(ma2_input).mean()
    
    # Create plotly object
    fig = go.Figure()
    
    # Plot closing prices
    fig.add_trace(go.Scatter(x=stockdata.index, y=stockdata.Close, name = "Closing Price"))
    
    # Plot trend line
    fig.add_trace(go.Scatter(x=stockdata.index, y=ma1, name=f"Moving Average: {ma1_input} days."))
    fig.add_trace(go.Scatter(x=stockdata.index, y=ma2, name=f"Moving Average: {ma2_input} days."))
    
    # Format layout (Show legend in top left corner, x axis slider,
    # remove title margins, remove backgroud colour,
    # label the axes
    fig.layout.update(showlegend = True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), 
                      xaxis_rangeslider_visible = True, margin=go.layout.Margin(l=60, r=0, b=0, t=30),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(title="Date"),yaxis=dict(title="Closing Price in USD"))
    
    # Show chart axes lines
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    
    # Show graph
    st.plotly_chart(fig, use_container_width = True)

def calculate_autocorrelation(stockdata):
    """
    This program calculates the amount of autocorrelation of a given stock
    to give additional insights for short-term traders. It is currently unused
    in the program but could be used for further analysis.
    """
    # Create matplotlib figure
    fig = plt.figure(figsize=(12,6))
    # Create autocorrelation plot
    ax = autocorrelation_plot(stockdata.Close)
    plt.title("Autocorrelation Values for Time Series")
    # Show plot
    st.pyplot(fig)


def plot_weighted_ma(stockdata, ticker, startdate, enddate):
    """
    A program that plots the ticker data over the given timeframe
    and provides a 6-day weighted moving average based on user input.
    """
    # Get number of days for the custom weighted average.
    # Days is set to a default 6 days.
    wa_days = 6
    
    # Define 3 columns for display in streamlit
    col1, col2, col3 = st.columns(3)
    
    # Get user input for custom weights - default values have been set - min/max has been set.
    weight1 = col1.number_input("Enter weight 1: ", min_value = 0.00, max_value = 1.00, value = 0.1)
    weight2 = col2.number_input("Enter weight 2: ", min_value = 0.00, max_value = 1.00, value = 0.1)    
    weight3 = col3.number_input("Enter weight 3: ", min_value = 0.00, max_value = 1.00, value = 0.1)
    weight4 = col1.number_input("Enter weight 4: ", min_value = 0.00, max_value = 1.00, value = 0.2)
    weight5 = col2.number_input("Enter weight 5: ", min_value = 0.00, max_value = 1.00, value = 0.2)            
    weight6 = col3.number_input("Enter weight 6: ", min_value = 0.00, max_value = 1.00, value = 0.3)
  
    # Transform weights into numpy array
    weights_for_ma = [weight1, weight2, weight3, weight4, weight5, weight6]
    weights_for_ma = np.array(weights_for_ma)
    
    # Check if weights make sense
    if sum(weights_for_ma) != 1.00:
        st.write(f"Current sum of weights: {sum(weights_for_ma)} - Please enter weights to equal 1")
    
    else:
        st.write("Weights have been set successfully!")
    
    # Create weighted MA data
    ma_weighted = stockdata.Close.rolling(wa_days).apply(lambda x: np.sum(weights_for_ma * x))
    
    # Create plotly object
    fig = go.Figure()
    
    # Plot closing prices
    fig.add_trace(go.Scatter(x=stockdata.index, y=stockdata.Close, name = "Closing Price"))
    
    # Plot trend line
    fig.add_trace(go.Scatter(x=stockdata.index, y=ma_weighted, name=f"Custom Weighted Moving Average"))

    # Format layout (Show legend in top left corner, x axis slider,
    # remove title margins, remove backgroud colour,
    # label the axes
    fig.layout.update(showlegend = True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), 
                      xaxis_rangeslider_visible = True, margin=go.layout.Margin(l=60, r=0, b=0, t=30),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(title="Date"),yaxis=dict(title="Closing Price in USD"))
    
    # Show chart axes lines
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
 
    # Show graph
    st.plotly_chart(fig, use_container_width = True)

def plot_macd(stockdata, ticker):
    """
    Code credit: https://www.alpharithms.com/calculate-macd-python-272222/
    This program plots the price chart combined with a moving average convergence / divergence.
    """


    # Calculate MACD values and append them to stockdata dataframe
    stockdata.ta.macd(close='Close', fast=12, slow=26, append=True)
    
    # Generate plotly plot object
    fig = make_subplots(rows=2, cols=1, subplot_titles=[f"Candlechart: Ticker {ticker} over time.", "MACD"],)
    
    # Append closing price line to first graph
    fig.append_trace(go.Scatter(x=stockdata.index, y=stockdata['Open'],line=dict(color='black', width=1),
        name='Open', legendgroup='1',), row=1, col=1)

    # Append candlesticks for first graph
    fig.append_trace(go.Candlestick(x=stockdata.index, open=stockdata['Open'], high=stockdata['High'], low=stockdata['Low'],
        close=stockdata['Close'], increasing_line_color='green', decreasing_line_color='red', showlegend=False), 
        row=1, col=1)
    
    # Append Fast Signal (%k) line to second graph
    fig.append_trace(go.Scatter(
        x=stockdata.index,
        y=stockdata['MACD_12_26_9'],
        line=dict(color='Blue', width=2),
        name='MACD',
        # showlegend=False,
        legendgroup='2',), row=2, col=1)
    
    # Append Slow signal (%d) line to second graph
    fig.append_trace(go.Scatter(
        x=stockdata.index,
        y=stockdata['MACDs_12_26_9'],
        line=dict(color='Orange', width=2),
        # showlegend=False,
        legendgroup='2',
        name='Signal'), row=2, col=1)
    
    # Colorize the data to emphasize difference between fast and slow signal
    colors = np.where(stockdata['MACDh_12_26_9'] < 0, 'red', 'green')
    
    # Append colorized histogram to second graph indicating difference between fast and slow signal
    fig.append_trace(go.Bar(x=stockdata.index, y=stockdata['MACDh_12_26_9'], name='Histogram', marker_color=colors), row=2, col=1)
    
    # Define layout for graphs: Font size and rangeslider.
    layout = go.Layout(font_size=14, margin=go.layout.Margin(l=60, r=0, b=0, t=30), xaxis=dict(title="Date"),yaxis=dict(title="Closing Price in USD"), xaxis_rangeslider_visible = False)
    
    # Update options and show plot
    fig.update_layout(layout)
    # Show graph
    st.plotly_chart(fig, use_container_width = True)