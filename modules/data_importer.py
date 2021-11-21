"""
Python Assignment: stock_data_importer.py
This program imports stock data based on user input and saves it as a Pandas Dataframe.
Currently, it is not used in the streamlit module.
It is used in the command line interface of this app.
"""

# Import relevant libraries
import pandas_datareader as data
import streamlit as st

# Get stock data from Yahoo Finance based on user input
def get_yahoo_data(stock_ticker, start_date, end_date):
    """
    This function gets Yahoo Finance Data based on a stock ticker and a given 
    time-frame and then saves that data as a Pandas Dataframe.
    """
    stock_data = data.DataReader(stock_ticker, "yahoo", start_date, end_date)
    return stock_data

def main():
    stock_data, stock_ticker, start_date, end_date = get_yahoo_data()
    
if __name__ == "__main__":
    main()