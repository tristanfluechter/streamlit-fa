"""
This is a stock evaluator and predictor that can be used to get information about
a certain stock over a user-defined timeframe. It uses descriptive statistics and 
web-crawling to give first insights into the stock - based on these statistics, the user can then
implement various predictive models - linear regression, ARIMA, LSTM and a sentiment analysis
to derive a decision on whether or not to buy the stock.

The app uses Streamlit to create a web-based GUI to enable easy hosting if desired.

The hosted app can be found at https://share.streamlit.io/tristanfluechter/streamlit-fa/main/stock_evaluator.py

Copyright: Tristan Fluechter, Odhrán McDonnell, Anirudh Bhatia, Kunal Gupta
"""

# Import necessary external libraries
import streamlit as st # to host the app
import datetime # to get correct date inputs
import pandas_datareader as data # to read stock data
import pickle # to import pre-trained ML countvector and randomforest

# Import necessary modules
import modules.descriptive_stats as ds # for descriptive statistics
import modules.financial_data as fd # for financial data
import modules.google_news as gn # to scrape google news
import modules.linear_regression as lr # to perform and evaluate linear regression
import modules.LSTM_prediction as lstm # to perform and evaluate LSTM
import modules.sentiment_analysis as sa # to perform and evaluate sentiment analysis

def homepage(stock_ticker, start_date, end_date):
    """
    This is the starting point of our module - the homepage.
    It performs no calculations.
    """
    
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
    Odhrán McDonnell,
    Anirudh Bhatia and
    Kunal Gupta   

    ***    
            
    """)

    
class StockPrediction:
    """
    The created class is the foundation of all further analysis. A StockPrediction
    instance incorporates closing price data, ticker, start & end date of analysis and
    stock news.
    
    The class methods will be used in the different tabs of the stock evaluator.
    """
    
    def __init__(self, stock_data, stock_ticker, start_date, end_date, stock_news):
        # Initialze the StockPrediction instance
        self.stock_data = stock_data
        self.stock_ticker = stock_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.stock_news = stock_news
    
    def stock_information(self):
        """
        Gets stock data based on ticker input in sidebar.
        Scrapes websites for Stock Information, Google News and Analyst Predictions.
        """
        
        # Header
        st.write("""
                 # Stock News & Analyst Predictions
                 
                 ***
                 
                 """)
        
        # Get trading volume information
        abs_change, trading_volume = fd.scrape_financial_kpi(self.stock_ticker)
        # Get Google News Headlines
        gn.print_headlines(self.stock_news, self.stock_ticker)
        # Spacer
        st.write("***")
        
        # Create two columns
        col1, col2 = st.columns([1,2])
        
        # Display column 1
        with col1:
            # Get current stock price
            current_stock_price = ds.show_stock_price(self.stock_ticker, abs_change, trading_volume)
            # Get analyst predictions and assign median_price
            median_price = fd.scrape_analyst_predictions(self.stock_ticker)
            
            # Logic check: Will stock price increase / decrease according to analysts?
            if current_stock_price < median_price:
                st.write("Analysts predict stock value will __increase__!")
            else:
                st.write("Analysts predict stock value will __decrease__!")
        
        # Display column 2    
        with col2:
            # Display historic closing price of stock over defined time period.
            st.subheader(f"Closing price of {self.stock_ticker} in USD over time:")
            ds.plot_stockdata(self.stock_data)
        
    def descriptive_stats(self):
        """
        Displays descriptive measures, trendline and simple moving average based on
        user input.
        """
        
        # Header
        st.write("""
                 # Descriptive Statistics
                 
                 ***
                 
                 """)
        
        # Create columns with relative sizes
        col1, col2, col3 = st.columns([2,1,4])
        
        # Column subheaders
        col1.subheader(f"Descriptive Measures")
        col2.subheader("Value")
        col3.subheader("Trendline based on Specified Timeframe")

        # Get descriptive measures
        descriptive_dict = ds.describe_stock_data(self.stock_data, self.stock_ticker)
        
        # Create descriptive measures table
        for key, value in descriptive_dict.items():
            col1.write(key)
            col2.write(value)
        
        # Plot trendline    
        with col3:
            ds.plot_trendline(self.stock_data)
        
        # Spacer
        st.write("***")
        
        # Display unweighted custom moving average based on user input
        st.subheader("Unweighted moving average - crossovers of MA lines indicate increase / decrease of stock value!")   
        ds.plot_simple_ma(self.stock_data)
    
    def advanced_descriptive_stats(self):
        """
        Displays custom-weighted 6-day moving average as well as MACD plot.
        """
        
        # Header
        st.write("""
            # Advanced Analytical Charts: Custom Moving Average and MACD
            To dive further into the stock analysis, this module provides you the option to examine
            a custom weighted moving average as well as an MACD graph.  
            ***
                 
            """)
        
        # Get user input for weights
        st.subheader("Enter weights to get custom 6-day weighted moving average!")
        # Plot weighted MA
        ds.plot_weighted_ma(self.stock_data, self.stock_ticker, self.start_date, self.end_date)
        # Plot MACD
        st.subheader("Candlechart Graph and Moving Average Convergence / Divergence")
        st.write("Crossing lines indicate stock downtrend / uptrend")
        ds.plot_macd(self.stock_data, self.stock_ticker)
    
    def prediction(self, stock_news):
        """
        Extensive module that uses regression, LSTM and Random Forest Decision Trees
        to predict stock value beahviour in the future.
        """
        
        # Header
        st.write("""
            # Stock Prediction: Linear Regression
            Statsmodels linear regression based on the obtained stock data,
            a user-specified timeframe and a desired prediction date. Please be aware that a simple regression
            is not sufficient to make a stock purchase decision.
            ***
                 
            """)
        
        # Get user input for target date and return prepared data for linear regression
        lr_target_date, lr_X, lr_Y = lr.linear_regression_dataprep(self.stock_data)
        # Create regression and return linear regression line and r_squared error
        lr_line, lr_rsquared = lr.linear_regression(self.stock_data, self.stock_ticker, lr_target_date, lr_X, lr_Y)
        # Evaluate predictive value of regression curve
        st.subheader("Evaluation of the Linear Regression Model for Predictive Use:")
        lr.linear_regression_evaluation(lr_Y, lr_line, lr_rsquared)
        
        # Header
        st.write("""
            # Stock Prediction: Long Short Term Memory
            LSTM is a form of time series prediction. The model is first trained based on a user-defined
            train-test split to avoid overfitting. The model can then make a prediction for the next 15 days.
            As the model auto-correlates with its own predictions over a longer timeframe, we advise using
            LSTM for short-term predictions only. Re-run the training by giving a new train-test-split.
            ***
                 
            """)
        
        # Prepare data for LSTM model: define how many days to consider, and create arrays with numerical value dates
        look_back, date_train, date_test, close_train, close_test, train_generator, test_generator, close_data_noarray, close_data = lstm.lstm_prepare_data(self.stock_data, self.stock_ticker)
        # Train LSTM model
        model, prediction, close_train, close_test = lstm.lstm_train(look_back, train_generator, test_generator, close_test, close_train)
        
        # Visualize model 
        st.subheader("Model Training: Does it fit the test data?")
        lstm.lstm_visualize(date_test, date_train, close_test, close_train, prediction, self.stock_ticker)
        # Visualize prediction
        st.subheader("Model Prediction for next 15 days")
        lstm.lstm_make_prediction(model, look_back, self.stock_data, close_data, close_data_noarray, self.stock_ticker)
        # Show evaluation of LSTM model
        st.subheader("Evaluation of the LSTM model for Predictive Use: ")
        lstm.lstm_evaluation(prediction, close_train)
        
        # Header
        st.write("""
            # Stock Prediction: Sentiment Analysis
            This module provides a sentiment analysis for a given stock ticker
            based on current Google News headlines. It has been trained on a dataset containing 6000 entries with
            25 headlines each. The model provides a positive or negative sentiment that can further help you
            make a purchase decision.
            ***
                 
            """)
        
        # Get countvector and randomclassifier from folder (pre-trained model!)
        countvector = pickle.load(open("data/vector.pickel", "rb"))
        randomclassifier = pickle.load(open("data/randomforest_sentiment_classifier.sav", "rb"))
        # Create prediction (0 or 1) for news headlines
        sa.rf_predict(stock_news, countvector, randomclassifier)


def get_stock_inputs():
    """
    Sidebar inputs for stock ticker and timeframe.
    Returns: stock_ticker, start_date, end_date
    """
    # TODO Error handling
    # TODO IF TICKER THEN NAVIGATION
    
    # Get ticker
    stock_ticker = st.sidebar.text_input("Please enter stock ticker:", value="MSFT")
    # Get start date (default date: Jan 1, 2021)
    start_date = st.sidebar.date_input("Please select a start date for stock analysis: ", value = datetime.date(2021,1,1))
    # Get end date (default date: Today)
    end_date = st.sidebar.date_input("Please select an end date for stock analysis: ", max_value= datetime.date.today())
    
    date_difference = (end_date - start_date).days
    
    if date_difference <= 60 and date_difference >= 0:
        st.sidebar.write(f"Selected timeframe is {date_difference} days. For optimal results, we recommend a timeframe > 60 days.")
    elif date_difference > 60:
        st.sidebar.write(f"Selected {date_difference} days timeframe.")
    else:
        st.sidebar.write("Selected timeframe is negative. Please enter in correct format.")

    return stock_ticker, start_date, end_date

# Cache function to save stock_data if nothing else has changed
@st.cache(allow_output_mutation=True)
def get_stock_data(stock_ticker, start_date, end_date):
    """
    Gets stock data from Yahoo based on sidebar inputs.
    Returns pd.DataFrame with all stock info.
    """
    
    stock_data = data.DataReader(stock_ticker, "yahoo", start_date, end_date)
    return stock_data

def app():
    """
    Main streamlit app to deploy.
    """
    
    # Set configuration to be widescreen
    st.set_page_config(layout="wide")
    
    # Get user input (needed for all functions!)
    try:
        stock_ticker, start_date, end_date = get_stock_inputs()
        
    except: 
        st.sidebar.write("Invalid stock ticker input. Please re-enter correct ticker.")
    
    # Get stock data if user input has been correct.
    # Error handling
    try: 
        stock_data = get_stock_data(stock_ticker, start_date, end_date)
        stock_news = gn.get_headlines(stock_ticker)
        st.sidebar.write(f"Successfully imported stock data for {stock_ticker}!")
    
    except:
        st.sidebar.write(f"Invalid ticker or date input. Please re-enter parameters.")
    
    # Create StockPrediction object
    try:
        stock = StockPrediction(stock_data, stock_ticker, start_date, end_date, stock_news)
    except:
        st.write("Could not create stock object. Please check if date and ticker has been entered correctly.")
    
    # Create navigation
    
    navigation = st.sidebar.selectbox(
        "Navigation",
        [
            "Homepage",
            "Basic Information",
            "Descriptive Statistics",
            "Advanced Analytical Charts",
            "Predictive Models"
        ],
    )

    # Navigate pages according to user input
    if navigation == "Homepage":
        homepage(stock_ticker, start_date, end_date)   
    elif navigation == "Basic Information":
        stock.stock_information()
    elif navigation == "Descriptive Statistics":
        stock.descriptive_stats()
    elif navigation == "Advanced Analytical Charts":
        stock.advanced_descriptive_stats()
    elif navigation == "Predictive Models":
        stock.prediction(stock_news)

# Run streamlit app
app()