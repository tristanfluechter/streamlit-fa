"""
This module provides a sentiment analysis for a given stock ticker
based on current Google News headlines.

Credit to: https://www.kaggle.com/rohit0906/stock-sentiment-analysis-using-news-headlines
All customizing has been done by the authors of this program.
"""

# Import relevant libraries
import pandas as pd # to handle dataframes
from sklearn.feature_extraction.text import CountVectorizer # to create word dict
from sklearn.ensemble import RandomForestClassifier # to classify word dict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # to report accuracy
import pickle # to load and save model
import regex as re # to replace characters
import os # to find cwd
import streamlit as st # to ensure streamlit useage


def train_rf_model():
    """
    Trains a randomforest decision tree with a dataset including stock news and the movement of the stock 
    market the next day.
    Dataset can be found at:  https://github.com/ronylpatil/Stock-Sentiment-Analysis
    
    Returns: countvector, randomclassifier
    """
    # Get current working directory
    cwd = os.getcwd()
    
    # Import base dataset. Labels: 0 == Stock going down, 1 == Stock going up.
    df = pd.read_csv(cwd + "/" "Stock News Training Dataset.csv", encoding = "ISO-8859-1")

    # Divide dataset into train and test
    train = df[df['Date'] < '20150101']
    test = df[df['Date'] > '20141231']

    print("Successfully imported news dataset!")

    # Remove special characters
    data = train.iloc[:, 2:27] # to only get the text columns
    data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

    # Create new columns for the 25 data columns
    list1 = [i for i in range(25)]
    new_Index=[str(i) for i in list1]
    data.columns = new_Index

    # convert everything to lowercase
    for index in new_Index:
        data[index] = data[index].str.lower()

    # create list to include all headlines into a single row
    headlines = []
    for row in range(0,len(data.index)):
        headlines.append(" ".join(str(x) for x in data.iloc[row, 0:25]))

    # implement BAG OF WORDS ML model
    countvector = CountVectorizer(ngram_range=(2,2))
    traindataset = countvector.fit_transform(headlines)
    vocab = countvector.vocabulary_

    # Save the countvector
    pickle.dump(countvector, open("vector.pickel", "wb"))

    # implement Random Forest Classifier
    randomclassifier = RandomForestClassifier(n_estimators=200, criterion='entropy')
    randomclassifier.fit(traindataset, train['Label'])

    # Predict for the Test Dataset
    test_transform = []
    for row in range(0,len(test.index)):
        test_transform.append(" ".join(str(x) for x in test.iloc[row, 2:27]))

    # transform dataset
    test_dataset = countvector.transform(test_transform)
    # make prediction
    predictions = randomclassifier.predict(test_dataset) # will predict a 0 or 1 outcome

    # how did the program do?
    # evaluate accuracy
    score = accuracy_score(test['Label'], predictions)
    st.write(f"Score of random forest classifier: {score}")
    # create report
    report = classification_report(test['Label'], predictions)
    st.write("Report of random classifier: ")
    st.write(report)

    # Save classifier
    filename = 'randomforest_sentiment_classifier.sav'
    pickle.dump(randomclassifier, open(filename, 'wb'))
    
    return countvector, randomclassifier

def rf_predict(stock_news, countvector, randomclassifier):

    # Create empty stock headlines list
    stock_news_headlines = []
    
    # Iterate through headlines and append
    for headline in range(len(stock_news)):
        stock_news_headlines.append(stock_news[headline]["title"])

    # Turn headlines into single string
    stock_news_headlines_temp = "".join(stock_news_headlines) 

    # Remove special characters
    stock_news_headlines = re.sub('[^A-Za-z0-9 ]+','',stock_news_headlines_temp)

    # Create list again to fit prediction format
    pred_headlines = ["".join(stock_news_headlines)] 

    # Transform headlines to fit random classifier
    dataset_to_predict = countvector.transform(pred_headlines)

    # Create prediction
    predictions = randomclassifier.predict(dataset_to_predict)

    if predictions == 1:
        st.write("Positive sentiment detected - buy stock.")
    else:
        st.write("Negative sentiment detected - sell stock.")
    
    return predictions