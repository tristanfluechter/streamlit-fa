"""
This is a stock evaluator and predictor that can be used to get information about
a certain stock over a user-defined timeframe. It uses descriptive statistics and 
web-crawling to give first insights into the stock - based on these statistics, the user can then
implement various predictive models - linear regression, ARIMA, LSTM and a sentiment analysis
to derive a decision on whether or not to buy the stock.

The app uses Streamlit to create a web-based GUI to enable easy hosting if desired.

Copyright: Tristan Fluechter, Odhrán MacDonnel, Anirudh Bhatia, Kunal Gupta
"""

# Import necessary external libraries
import streamlit as st
import datetime
import pandas_datareader as data
import pickle

# Import necessary modules
import modules.data_importer as data_importer
import modules.descriptive_stats as ds
import modules.financial_data as fd
import modules.google_news as gn
import modules.linear_regression as lr
import modules.LSTM_prediction as lstm
import modules.sentiment_analysis as sa

def homepage(stock_ticker, start_date, end_date):
    # Welcome page for the user
    # Display header image
    
    st.image('images/Welcome_Page.jpg')

    # Display greeting message
    st.write("""
    # Programming Assignment: Stock Evaluator
    This is a stock evaluation program that provides information about a certain stock ticker
    over a user-specified timeframe. The statistics and predictive models used will allow you
    to make a data-based decision on whether to buy a stock. Navigate the app using the sidebar navigation.
    
    Please enter your stock ticker and desired timeframe in the sidebar widgets!
    
    __This app has been created by:__
    
    Tristan Fluechter,
    Odhrán MacDonnel,
    Anirudh Bhatia and
    Kunal Gupta   

    ***    
            
    """)

    
class StockPrediction:
    
    def __init__(self, stock_data, stock_ticker, start_date, end_date, stock_news):
        self.stock_data = stock_data
        self.stock_ticker = stock_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.stock_news = stock_news
    
    def stock_information(self):
        st.write("""
                 # Stock News & Analyst Predictions
                 
                 ***
                 
                 """)
        
        abs_change, percent_change, trading_volume = fd.scrape_financial_kpi(self.stock_ticker)
        
        gn.print_headlines(self.stock_news, self.stock_ticker)
        st.write("***")
        
        col1, col2 = st.columns([1,2])
        
        with col1:
            current_stock_price = ds.show_stock_price(self.stock_ticker, abs_change)
            median_price = fd.scrape_analyst_predictions(self.stock_ticker)
            if current_stock_price < median_price:
                st.write("Analysts predict stock value will __increase__!")
            else:
                st.write("Analysts predict stock value will __decrease__!")
            
        with col2:
            st.subheader(f"Closing price of {self.stock_ticker} in USD over time:")
            ds.plot_stockdata(self.stock_data)
        
    def descriptive_stats_1(self):
        st.write("""
                 # Descriptive Statistics
                 
                 ***
                 
                 """)
        
        col1, col2, col3 = st.columns([2,1,4])
        
        col1.subheader(f"Descriptive Measures for {self.stock_ticker}")
        col2.subheader(" Value")
        col3.subheader("Trendline based on Specified Timeframe")
    
        descriptive_dict = ds.describe_stock_data(self.stock_data, self.stock_ticker)
        
        for key, value in descriptive_dict.items():
            col1.write(key)
            col2.write(value)
            
        with col3:
            ds.plot_trendline(self.stock_data)
        st.write("***")
        st.subheader("Unweighted moving average - crossovers of MA lines indicate increase / decrease of stock value!")   
        ds.plot_simple_ma(self.stock_data)
    
    def descriptive_stats_2(self):
        st.write("""
            # Advanced Analytical Charts: Custom Moving Average and MACD
            To dive further into the stock analysis, this module provides you the option to examine
            a custom weighted moving average as well as an MACD graph.  
            ***
                 
            """)
        st.subheader("Enter weights to get custom 6-day weighted moving average!")
        ds.plot_weighted_ma(self.stock_data, self.stock_ticker, self.start_date, self.end_date)
        st.subheader("Candlechart Graph and Moving Average Convergence / Divergence")
        st.write("Crossing lines indicate stock downtrend / uptrend")
        ds.plot_macd(self.stock_data, self.stock_ticker, self.start_date, self.end_date)
        st.subheader("Autocorrelation Plot: How is the time series correlated with itself?")
        ds.calculate_autocorrelation(self.stock_data)
    
    def prediction(self, stock_news):
        st.write("""
            # Stock Prediction: Linear Regression
            Statsmodels linear regression based on the obtained stock data,
            a user-specified timeframe and a desired prediction date. Please be aware that a simple regression
            is not sufficient to make a stock purchase decision.
            ***
                 
            """)
        lr_target_date, lr_X, lr_Y = lr.linear_regression_dataprep(self.stock_data)
        lr_line, lr_squared = lr.linear_regression(self.stock_data, self.stock_ticker, lr_target_date, lr_X, lr_Y)
        st.subheader("Evaluation of the Linear Regression Model for Predictive Use:")
        lr.linear_regression_evaluation(lr_Y, lr_line, lr_squared)
        
        st.write("""
            # Stock Prediction: Long Short Term Memory
            LSTM is a form of time series prediction. The model is first trained based on a user-defined
            train-test split to avoid overfitting. The model can then make a prediction for the next 15 days.
            As the model auto-correlates with its own predictions over a longer timeframe, we advise using
            LSTM for short-term predictions only. Re-run the training by giving a new train-test-split.
            ***
                 
            """)
        
        look_back, date_train, date_test, close_train, close_test, train_generator, test_generator, close_data_noarray, close_data = lstm.lstm_prepare_data(self.stock_data, self.stock_ticker)
        model, prediction, close_train, close_test = lstm.lstm_train(look_back, train_generator, test_generator, close_test, close_train)
        
        st.subheader("Model Training: Does it fit the test data?")
        lstm.lstm_visualize(date_test, date_train, close_test, close_train, prediction, self.stock_ticker)
        st.subheader("Model Prediction for next 15 days")
        lstm.lstm_make_prediction(model, look_back, self.stock_data, close_data, close_data_noarray, self.stock_ticker)
        
        st.subheader("Evaluation of the LSTM model for Predictive Use: ")
        lstm.lstm_evaluation(prediction, close_train)

        st.write("""
            # Stock Prediction: Sentiment Analysis
            This module provides a sentiment analysis for a given stock ticker
            based on current Google News headlines. It has been trained on a dataset containing 6000 entries with
            25 headlines each. The model provides a positive or negative sentiment that can further help you
            make a purchase decision.
            ***
                 
            """)
        countvector = pickle.load(open("data/vector.pickel", "rb"))
        randomclassifier = pickle.load(open("data/randomforest_sentiment_classifier.sav", "rb"))
        sa.rf_predict(stock_news, countvector, randomclassifier)


def get_stock_inputs():
    
    # Get ticker
    stock_ticker = st.sidebar.text_input("Please enter stock ticker:", value="MSFT")
    # Get start date (default date: Jan 1, 2021)
    start_date = st.sidebar.date_input("Please select a start date for stock analysis: ", value = datetime.date(2021,1,1))
    # Get end date (default date: Today)
    end_date = st.sidebar.date_input("Please select an end date for stock analysis: ")

    return stock_ticker, start_date, end_date

@st.cache(allow_output_mutation=True)
def get_stock_data(stock_ticker, start_date, end_date):
    
    stock_data = data.DataReader(stock_ticker, "yahoo", start_date, end_date)
    return stock_data

def app():
    
    st.set_page_config(layout="wide")
    
    stock_ticker, start_date, end_date = get_stock_inputs()
    
    
    try: 
        stock_data = get_stock_data(stock_ticker, start_date, end_date)
        stock_news = gn.get_headlines(stock_ticker)
        st.sidebar.write(f"Successfully imported stock data for {stock_ticker}!")
    
    except:
        st.sidebar.write(f"Invalid ticker or date input. Please re-enter parameters.")
    
    stock = StockPrediction(stock_data, stock_ticker, start_date, end_date, stock_news)
    
    navigation = st.sidebar.selectbox(
        "Navigation",
        [
            "Homepage",
            "Basic Information",
            "Descriptive Statistics",
            "Advanced Analytical Charts",
            "Price Prediction"
        ],
    )

    if navigation == "Homepage":
        homepage(stock_ticker, start_date, end_date)
        
    elif navigation == "Basic Information":
        stock.stock_information()
    elif navigation == "Descriptive Statistics":
        stock.descriptive_stats_1()
    elif navigation == "Advanced Analytical Charts":
        stock.descriptive_stats_2()
    elif navigation == "Price Prediction":
        stock.prediction(stock_news)

  
app()  
 