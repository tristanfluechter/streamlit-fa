"""
This module provides the user with a basic linear regression to make a
prediction for a given date.
"""
    
# Import relevant libraries
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
import statsmodels.api as sm
import datetime as dt
import streamlit as st
import plotly.graph_objects as go


def linear_regression_dataprep(stockdata):
    """
    This program performs a linear regression based on the obtained stock data,
    a user-specified timeframe and a desired prediction date.
    It outputs a graph that includes all stated information.
    
    Returns: lr_target_date, lr_X, lr_Y
    """
    
    # Get length of dataframe (ie number of days we can consider for linear regression)
    stockdata_days = len(stockdata)
    

    col1, col2 = st.columns(2)
    
    # Get user input for the prediction date
    # Ensure that target date is in the future
    lr_target_date = col1.date_input("Please enter your target date you want to predict the price for: ", 
                                     min_value = dt.date.today(), value = dt.date(2021,12,24)) 
    
    # Ask user how many days before current date to use for linear regression
    # Ensure that amount of days does not exceed dataframe length
    lr_days = col2.number_input(f"How many past days do you want to consider for the linear regression? (max: {str(stockdata_days)}) ",
                                min_value = 15, max_value=stockdata_days, value=60) # 
    
    # Create dataset with the last lr_days days
    lr_dataframe = stockdata.tail(lr_days)
       
    # Change datetime to ordinal data for linear regression and define X & Y
    lr_X = np.asarray(lr_dataframe.index.map(dt.datetime.toordinal))
    lr_Y = np.asarray(lr_dataframe.Close)
 
    # Reshape Data into Numpy Array
    lr_X = lr_X.reshape(-1,1)
    lr_Y = lr_Y.reshape(-1,1)
    
    return lr_target_date, lr_X, lr_Y

def linear_regression(stockdata, ticker, targetdate, lr_X, lr_Y):
    """
    This program creates a linear regression with the preprocessed data
    to make a prediction based on a user-input target date.
    
    Returns: lr_line, lr_squared
    """
        
    # Create statsmodels LR object and add lr_X
    x = sm.add_constant(lr_X)
        
    # Predict Results
    lr_results = sm.OLS(lr_Y,x).fit() # variables do not need to be scaled for OLS regression.
        
    # Give summary of linear regression
    lr_results.summary()
        
    # Assign linear regression curve y-intercept and slope based on summary table
    lr_slope = lr_results.summary2().tables[1]["Coef."][1] # to get slope coefficient
    lr_y_intercept = lr_results.summary2().tables[1]["Coef."][0] # to get intercept
    lr_rsquared = lr_results.summary2().tables[0][3][0]
        
    # Convert user input date to ordinal value
    lr_target_date = targetdate.toordinal()
        
    # Append target date to lr_X and lr_Y
    lr_X = np.append(lr_X, lr_target_date)
    lr_Y = np.append(lr_Y, (lr_target_date * lr_slope + lr_y_intercept))
        
    # Create linear regression dataset to plot later
    lr_line = lr_X * lr_slope + lr_y_intercept
        
    # Retransform dates to datetime to create useful x-axis
    # Create shape var to find out current state of lr_X
    x_shape = lr_X.shape
        
    # Create list object from np_array (lr_X)
    lr_X_list = lr_X.reshape(1, x_shape[0])[0].tolist()
        
    # Create list of dates with iteration
    dates = []

    for number in lr_X_list:
        created_date = dt.date.fromordinal(number)
        dates.append(created_date)
    
    # Create plotly object
    fig = go.Figure()
    
    # Plot closing prices
    fig.add_trace(go.Scatter(x=dates, y=lr_Y, mode = "markers", name = "Closing Price"))
    
    # Plot Regression Line for existing values
    fig.add_trace(go.Scatter(x=dates[:-1], y=lr_line[:-1], name="Regression Line",line=dict(color="red"), mode = "lines"))
    # Plot Regression Line prediction
    fig.add_trace(go.Scatter(x=dates[-2:], y=lr_line[-2:], name=f"Regression Line Prediction", line=dict(color="red", dash="dash")))
    
    # Format layout (Show legend in top left corner,
    # remove title margins, remove backgroud colour,
    # label the axes
    fig.layout.update(showlegend = True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), 
                      margin=go.layout.Margin(l=60, r=0, b=0, t=30),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(title="Date"),yaxis=dict(title="Closing Price in USD"))
    
    # Format axes
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    
    # Add annotation to prediction date
    fig.add_annotation(x=dates[-1], y=(lr_line[-1]+1),
            text="Predicted Date")
    
    # Show plot
    st.plotly_chart(fig, use_container_width=True)
    
    return lr_line, lr_rsquared

def linear_regression_evaluation(lr_Y, lr_line, lr_rsquared):
    """
    Evaluates the linear regression accuracy and the level at which it
    can give insights into the stock price movement.
    """
    # Calculate RMSE based on lr_line (regression) and lr_y (actual values)
    root_mean_square_error = np.sqrt(((lr_line - lr_Y) ** 2).mean()).round(2)
    # RSquared was returned in the Statsmodels OLS Regression summary
    r_squared = float(lr_rsquared)
    
    # Print out results
    st.write(f"Linear Regression RMSE: {root_mean_square_error}")
    st.write(f"Linear Regression R-Squared: {r_squared}")
    
    # Evaluate r-squared metric - how much of the movements does the regression explain?
    if r_squared <= 0.4:
        st.write(f"With an r-squared value of {r_squared}, it is __not sufficient__ to rely on a simple regression to predict stock values.")
    else:
        st.write(f"With an r-squared value of {r_squared}, the regression seems to __identify a trend in stock prices__. However, we advise to use additional predictive measures.")
    

def main():
    lr_target_date, lr_X, lr_Y = linear_regression_dataprep()
    lr_line, lr_rsquared = linear_regression()
    linear_regression_evaluation(lr_Y, lr_line, lr_rsquared)

if __name__ == "__main__":
    main()