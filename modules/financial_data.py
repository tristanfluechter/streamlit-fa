"""
This module scrapes financial websites to obtain and output
financial data such as analyst predictions and key financial KPI.
It uses different websites to obtain this information since some website don't make the
presented values easily available.
"""

# Import relevant libraries
from bs4 import BeautifulSoup # html parser
import requests # web scraper
import re # regex to search for strings on sites
import streamlit as st

def scrape_analyst_predictions(ticker):
    """
    This program uses beautiful soup to parse the CNN money page
    and returns the analyst predictions for a given stock ticker
    """
    # Get the CNN Money Website for a given stock ticker
    URL = f"https://money.cnn.com/quote/forecast/forecast.html?symb={ticker}"
    page = requests.get(URL)

    # Parse the website with beautifulsoup
    soup = BeautifulSoup(page.content, "html.parser")

    # The relevant website information has been isolated with "inspect element"
    # Find the Analysts Offering string in the parsed webpage
    find_string = soup.body.findAll(text=re.compile('analysts offering'), limit=1) # returns a list
    
    # Extract the string form find_string
    analyst_predictions_text = find_string[0]
    
    # Search and extract the median price prediction, highest and lowest price prediction.
    analyst_predictions = re.findall("\d+\.\d+", analyst_predictions_text)
    
    # Extract price information
    median_price = float(analyst_predictions[0])
    high_price = float(analyst_predictions[1])
    low_price = float(analyst_predictions[2])
    
    return median_price, high_price, low_price
    
def scrape_financial_kpi(ticker):
    """
    This program uses beautiful soup to parse the Marketwatch
    website to retrieve key financial KPI.
    """
    
    # Get the Market Watch website for a given stock ticker
    URL = f"https://www.marketwatch.com/investing/stock/{ticker}/financials"
    page = requests.get(URL)

    # Parse the website with beautifulsoup
    soup = BeautifulSoup(page.content, "html.parser")
    
    # Find the Analysts Offering string in the parsed webpage
    # Intraday change is either stored in a "positive" or "negative" table cell
    span_change_intraday = soup.body.find_all('span', {'class':"change--point--q"}) 
  
    # Find absolute change and percent change in obtained data
    abs_change = re.findall("[-]?\d+\.\d+", str(span_change_intraday[0]))[1] # first value in table is abs change
    
    # Find current trading volume
    span_trading_volume = soup.body.find_all('span', {'class':"primary"})
    trading_volume = re.findall("\d+\.\d+M", str(span_trading_volume[1]))[0]
    
    # Print out intraday change information
    return abs_change, trading_volume
    
    
def main():
    scrape_analyst_predictions("AAPL")
    scrape_financial_kpi("AAPL")

if __name__ == "__main__":
    main()