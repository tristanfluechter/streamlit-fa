"""
This module uses Facebook Prophet to create a prediction based on the given
time series. Credits to the code: https://www.kaggle.com/ahmetax/fbprophet-and-plotly-example
All customising has been done by the authors of this program.
"""
# Import necessary modules
import data_importer
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

import warnings
warnings.filterwarnings('ignore')  # Hide warnings
import plotly.graph_objects as go
import matplotlib.pyplot as plt

stock_data = data_importer.get_yahoo_data("MSFT", "2020-01-01", "2021-10-10")



def prophet_forecast(stock_data):
    # Reset Index
    stock_data.reset_index(inplace=True)
    
    # Get relevant columns
    prophet_data = stock_data[["Date","Adj Close"]]

    # Rename columns
    prophet_data = prophet_data.rename(columns={"Date": "ds", "Adj Close": "y"})

    # Define train-test-split
    split = int(len(prophet_data) * 0.6)

    # Split data
    prophet_data_train = prophet_data[0:split]
    prophet_data_test = prophet_data[split:]

    # Create Prophet Model
    m = Prophet()
    m.fit(prophet_data_train)

    # Make future predictions
    future = m.make_future_dataframe(periods=411)
    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    # Create plotly figure
    fig = plot_plotly(m, forecast)
    
    # Format with program-specific Layout
    # remove title margins, remove backgroud colour,
    # label the axes
    fig.layout.update(showlegend = True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), 
                      margin=go.layout.Margin(l=60, r=0, b=0, t=30), xaxis_rangeslider_visible = False,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(title="Date"),yaxis=dict(title="Closing Price in USD"))
    
    # Show chart axes lines
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    
    fig.data[0].name="Actual Price"
    fig.data[1].name="Lower Bound"
    fig.data[3].name="Upper Bound"
    
    fig.show()
    
    fig2 = plot_components_plotly(m, forecast)
    fig2.show()

prophet_forecast(stock_data)