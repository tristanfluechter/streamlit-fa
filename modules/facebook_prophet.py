"""
This module uses Facebook Prophet to create a prediction based on the given
time series. Credits to the code: https://www.kaggle.com/ahmetax/fbprophet-and-plotly-example
All customising has been done by the authors of this program.
"""
# Import necessary modules
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
# import warnings
# warnings.filterwarnings('ignore')  # Hide warnings
import plotly.graph_objects as go
import streamlit as st
pd.core.common.is_list_like = pd.api.types.is_list_like
# import data_importer

# stock_data = data_importer.get_yahoo_data("AAPL", "2020-01-01", "2021-01-01")

def prophet_forecast(stock_data):
    """
    This module creates a stock forecast with the Facebook Prophet model based on the
    stock_data dataframe. It also creates a trendline as well as a weekday analysis of
    stock movement.
    """
    
    # Reset Dataframe Index
    stock_data["Date"] = stock_data.index
    
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
    
    # Get last date of prediction 
    prophet_pred = forecast["Trend"].iloc[-1]

    # Create plotly figure for forecast
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
    
    # Rename legend (Prophet naming unsuitable)
    fig.data[0].name="Actual Price"
    fig.data[1].name="Lower Bound"
    fig.data[3].name="Upper Bound"
    
    # Subheader
    st.subheader("Facebook Prophet Price Prediction")
    
    # Show prediction graph
    st.plotly_chart(fig, use_container_width = True)
    
    # Create components analysis plotly object
    fig2 = plot_components_plotly(m, forecast)
    
    # Formate layout and axes
    fig2.layout.update(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                       margin=go.layout.Margin(l=60, r=0, b=0, t=30))
    
    fig2.update_yaxes(title_text="Trend", row=1)
    fig2.update_yaxes(title_text="Weekday Trend", row=2)
    
    # Show chart axes lines
    fig2.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig2.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    
    # Subheader
    st.subheader("Facebook Prophet Components Analysis")
    
    # Show Graph
    st.plotly_chart(fig2, use_container_width = True)

    return prophet_pred