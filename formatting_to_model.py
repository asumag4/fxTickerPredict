# Imports 

import pandas as pd
import numpy as np 
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import scipy

from prophet import Prophet
from json.decoder import JSONDecodeError

import yfinance as yf 

import tensorflow

import json

from pathlib import Path


# --- Import the data --- 
def get_raw_article_data(ticker):
    file_name_in_dir = f"/{ticker}_fxstreet_database"
    articles = pd.read_parquet(r"data/raw_scrapped_data" + file_name_in_dir)
    # Immediately convert `date` column to datetime
    articles['date'] = articles['date'].apply(convert_to_datetime)
    return articles

# --- Import the current modelling dataset --- 
def get_current_modelling_data(ticker):
    file_name_in_dir = f"/{ticker}_modelling_dataset"
    articles = pd.read_parquet(r"data/modelling_dataset" + file_name_in_dir)
    return articles

# --- Initialize finBERT --- 
def init_finBert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, finbert

# --- Helper Function: determine sentiment polarity --- 
def analyze_sentiment(tokenizer, finbert, x):

    # Determine how BERT will analyze the text
    tokenizer_kwargs = {
    "padding" : True,
    "truncation" : True,
    "max_length" : 512,
    }

    with torch.no_grad():
        input_sequence = tokenizer(x, return_tensors="pt", **tokenizer_kwargs)
        logits = finbert(**input_sequence).logits
        # Apply softmax to get probabilities 
        prob = scipy.special.softmax(logits.numpy().squeeze())
        weights = {
            0 : -1, # Negative
            1 : 0, # Neutral
            2 : 1 # Positive
        }
        # Calculate polarity score
        polarity = sum(prob[i] * weights[i] for i in range(len(prob)))
    return polarity # Returns raw sentiment scores (as probabilities)

# --- Helper Function: Converting Obj to Datetime 
def convert_to_datetime(date_str):
    date_format = "%m/%d/%Y %H:%M:%S GMT"
    return datetime.strptime(date_str, date_format)

# --- Helper Function: Logging most current news data
def log_most_current_news(data, ticker, log):
    
    # Log the most recent article now
    data = data.sort_values(by='date', ascending=False)
    last_logged_time = data['date'].iloc[0]

    # Convert to a datetime object if needed
    if isinstance(last_logged_time, str):
        last_logged_time = convert_to_datetime(last_logged_time)
    
    last_logged_time_str = last_logged_time.strftime("%m/%d/%Y %H:%M:%S GMT")

    log[ticker] = last_logged_time_str

    with open("data/modelling_dataset/log.json", mode="+w") as file:
        json.dump(log, file, indent=4)

# --- Helper Function: Reading `log.json` file --- 
def read_log_file(): 
    with open("data/modelling_dataset/log.json", mode="r") as file:
        log = json.load(file)
    return log


# --- Getting average daily news sentiment ---

def get_high_low_daily_val(start, end, ticker):
    ticker_data = ticker.history(start=start, end=end, interval="1d").reset_index()
    return float(ticker_data.loc[0,"High"]), float(ticker_data.loc[0,"Low"])

# Function for finding daily ticker values; high and low 
def find_high_low_daily_val(day, yf_ticker):
    following_day = day + timedelta(days=1)
    try:
        return get_high_low_daily_val(day, following_day, yf_ticker)
    except:
        try: # For when posted on a Saturday, go one day back 
            previous_day = day - timedelta(days=1)
            return get_high_low_daily_val(previous_day, day, yf_ticker)
        except: 
            try: # For when posted on a sunday, go one day back 
                previous_2_days = day - timedelta(days=2)
                return get_high_low_daily_val(previous_2_days, previous_day, yf_ticker)
            except:
                return None, None

def format_articles_to_daily(articles, ticker): 
    # Get the aggregated daily sentiment
    daily_sentiment = articles[['date','sentiment_polarity']].groupby([articles['date'].dt.date]).mean('sentiment_polarity').reset_index()


    map_yf_ticker = {
        "EURUSD=X" : yf.Ticker("EURUSD=X"),
        "JPY=X" : yf.Ticker("JPY=X"),
        "GBPUSD=X" : yf.Ticker("GBPUSD=X"),
    }

    yf_ticker = map_yf_ticker[ticker]

    # Add in daily high and low data 
    daily_sentiment[['High','Low']] = pd.DataFrame(daily_sentiment['date'].apply(lambda day: find_high_low_daily_val(day, yf_ticker)).tolist(), index=daily_sentiment.index)

    print("--------------------------------------------")
    print(f"The number of null 'High' and 'Low' values found  \n {daily_sentiment.isnull().sum()}")
    print("--------------------------------------------")
    daily_sentiment.dropna(inplace=True)

    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])

    return daily_sentiment

def save_modelling_data(data, ticker):
    file_name = f"/{ticker}_modelling_dataset"
    location = r"data/modelling_dataset"
    data.to_parquet(location + file_name)
    return

def generate_modelling_data(data, ticker):

    tokenizer, finbert = init_finBert()

    # Get the sentiment polarity
    data['sentiment_polarity'] = data['text'].apply(lambda x: analyze_sentiment(tokenizer=tokenizer, finbert=finbert, x=x))
    print("Successfully retrieved the sentiment")

    # Convert the raw formatted data to aggregated daily data 
    data = format_articles_to_daily(data, ticker)
    print("Successfully aggregated the date into daily format")

    print(f"Final modelling dataset: {data.info}")

    return data

# --- Helper function: Generating new database 
def generate_fresh_database(data, ticker, log={}):

    # Get the raw data 
    data = get_raw_article_data(ticker)

    # Log the most recent article now
    log_most_current_news(data, ticker, log)

    # Format the raw data
    data = generate_modelling_data(data, ticker)

    # Save the data
    save_modelling_data(data, ticker)

# --- Main ---
def update_modelling_data(ticker):

    # Retrieve `log.json`
    logging_file = Path("data/modelling_dataset/log.json")
    modelling_data = Path(f"data/modelling_dataset/{ticker}_modelling_dataset")

    print(f"Working on {ticker} raw data!")

    # Get the last date article the modelling data set was updated on, and retrieve work on articles AFTER the last logged article
    try: 
        # If `log.json` even exists..
        if (logging_file.is_file() and modelling_data.is_file()):
            print(f"Logging file and current modelling dataset exists for {ticker}")

            # Get the raw data 
            raw_data = get_raw_article_data(ticker)

            # Opening `log.json` 
            log = read_log_file()

            # Find the last logged date
            last_logged_time = log[ticker]
            last_logged_time = datetime.strptime(last_logged_time, "%m/%d/%Y %H:%M:%S GMT")

            # Check if the most recent article is also the last logged date, if so skip this process..
            raw_data = raw_data.sort_values(by='date', ascending=False)
            most_recent_article_time = raw_data['date'].iloc[0]
            if (most_recent_article_time == last_logged_time):
                print(f"The last logged article was the same as the most recent article scraped... skipping {ticker}")
                return

            # IF NOT:

            # Get the articles that are after the last logged article
            data = raw_data[raw_data['date'] > last_logged_time]

            # Report to the user
            print(f"The last logged article was published on: {last_logged_time}")
            print("Now updating the modelling dataset")

            log_most_current_news(data, ticker, log)
            # With only the articles we need to work with; generate the modelling data from that
            data = generate_modelling_data(data, ticker)

            # Open up the modelling dataset 
            modelling_data = get_current_modelling_data(ticker) 

            # Combine data & modelling dataset
            data = pd.concat([data, modelling_data], axis=0)

            # If there are rows with duplicate `date` data (there should only be 2 rows if ever)
            same_day = data[data.duplicated(subset=['date'], keep=False)]

            if (not same_day.empty):
                print("Found days with the same day!")
                print(same_day)
                # We can reuse `format_articles_to_daily` to ensure we get the correct, most up to date
                consolidated_row = format_articles_to_daily(same_day, ticker)

                # We'll remove these duplicates
                data = data[~data.duplicated(subset=['B'], keep=False)]

            # And then append our consolidated row 
            data = pd.concat([consolidated_row, data], axis = 0)

            # Then save your data
            save_modelling_data(data, ticker)

        elif (logging_file.is_file() and (not modelling_data.is_file())):

            # The file exists, so just append the new ticker symbol
            print(f"Logs do not exists but the modelling data for {ticker} does, creating new log and modelling data for {ticker}")

            log = read_log_file()

            data = get_raw_article_data(ticker)

            generate_fresh_database(data, ticker, log)

        else: 
            print("Logging file doesn't exist, initiating one!")
            print(f"Logs do not exists but the modelling data for {ticker} doesn't, creating new log and modelling data for {ticker}")

            data = get_raw_article_data(ticker)

            generate_fresh_database(data, ticker)

    except JSONDecodeError:
        print("Log file is corrupted or empty, resetting log!")
        print(f"Logs do not exists but the modelling data for {ticker} doesn't, creating new log and modelling data for {ticker}")

        data = get_raw_article_data(ticker)

        generate_fresh_database(data, ticker)

# --- RUNNING THE PROGRAM --- 

# Create a ticker for all currency pairs
EURUSD = "EURUSD=X" # EUR/USD
USDJPY = "JPY=X"
GBPUSD = "GBPUSD=X"

for ticker in [EURUSD, USDJPY, GBPUSD]:
    update_modelling_data(ticker)