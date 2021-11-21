"""
This module provides a sentiment analysis for a given stock ticker
based on current Google News headlines.

Credit to: https://www.kaggle.com/rohit0906/stock-sentiment-analysis-using-news-headlines
All customizing has been done by the authors of this program.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
from gnews import GNews
import regex as re
import os
import streamlit as st


def train_rf_model():
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

    test_dataset = countvector.transform(test_transform)
    predictions = randomclassifier.predict(test_dataset) # will predict a 0 or 1 outcome

    # how did the program do?
    score = accuracy_score(test['Label'], predictions) # tests score
    st.write(f"Score of random forest classifier: {score}")
    report = classification_report(test['Label'], predictions)
    st.write("Report of random classifier: ")
    st.write(report)

    # Save classifier
    filename = 'randomforest_sentiment_classifier.sav'
    pickle.dump(randomclassifier, open(filename, 'wb'))
    
    return countvector, randomclassifier

def rf_predict(stock_news, countvector, randomclassifier):

    # Get headlines from JSON object
    stock_news_headlines = []
    
    # Iterate through headlines
    for headline in range(len(stock_news)):
        stock_news_headlines.append(stock_news[headline]["title"])

    stock_news_headlines_temp = "".join(stock_news_headlines) # headlines are now strings

    stock_news_headlines = re.sub('[^A-Za-z0-9 ]+','',stock_news_headlines_temp)

    pred_headlines = ["".join(stock_news_headlines)] # join the list together as a string, then turn it into a list again to remove commas

    dataset_to_predict = countvector.transform(pred_headlines)

    predictions = randomclassifier.predict(dataset_to_predict)

    if predictions == 1:
        st.write("Positive sentiment detected - buy stock.")
    else:
        st.write("Negative sentiment detected - sell stock.")