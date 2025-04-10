# Imports 

import pandas as pd
import numpy as np 
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import scipy

from json.decoder import JSONDecodeError

import yfinance as yf 

import json

# ------ Additional Functions to Adapt Program to AWS Environment ------

# AWS Env Adapt: 
import boto3
import io

import pyarrow as pa
import pyarrow.parquet as pq

# --- Helper function: Reading .parquet files from S3 Buckets ---  
def read_s3_parquet(Bucket, Key):
    response = S3_CLIENT.get_object(
        Bucket=Bucket,
        Key=Key
    )
    # Read the binary content from the S3 object
    content = response['Body'].read()
    # Load the parquet file directly in to a pandas dataframe
    data = pd.read_parquet(io.BytesIO(content))
    return data

def write_s3_parquet(Bucket, data, ticker):
    # This function is to write a the modelling_dataset to parquet dataframe to S3 as a parquet file
    Key = f"modelling_dataset/{ticker}_modelling_dataset"

    buffer = io.BytesIO()
    table = pa.Table.from_pandas(data)
    pq.write_table(table, buffer)

    try:
        # Upload the parquet file from memory to S3
        S3_CLIENT.put_object(
            Bucket=Bucket,
            Key=Key,
            Body=buffer.getvalue()
        )
        print(f"{ticker} modelling dataset successfully uploaded as parquet to 's3://{Bucket}/{Key}")
    except Exception as e:
        print(f"Error uploading {ticker} modelling dataset to S3: {e}")

# --- Helper function: Writing log.json to S3 --- 
def write_log_to_s3(log):
    # Convert the dict to a JSON string
    log_json = json.dumps(log, indent=4)

    # Upload the JSON string to S3
    S3_CLIENT.put_object(
        Bucket=MODELLING_DATA_BUCKET,
        Key="modelling_dataset/log.json",
        Body=log_json
    )
    print(f"Successfully written log.json to S3 Bucket {MODELLING_DATA_BUCKET} as 'log.json'.")

# --- Helper function: reading log.json from S3 bucket
def read_log_from_s3():
    try: 
        # Get the file from S3
        response = S3_CLIENT.get_object(
            Bucket=MODELLING_DATA_BUCKET,
            Key="modelling_dataset/log.json"
        )

        # Read and decode the file content
        log_json = response['Body'].read().decode('utf-8')

        # Convert the JSON string to a Python dictionary
        log = json.loads(log_json)
        print("Successfully read log.json from S3 Bucket.")
        return log
    except:
        print("Error loading log.json from S3 Bucket.")
        return None
    
# --- Helper Function: Function to check if a file exists in S3 ---
def file_exists_in_s3(bucket_name, s3_key):
    try:
        S3_CLIENT.head_object(Bucket=bucket_name, Key=s3_key)
        return True
    except S3_CLIENT.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise

# ------ Regular Functions ------

# --- Import the data --- 
def get_raw_article_data(ticker):
    file_name_in_dir = f"/{ticker}_fxstreet_database"
    articles = read_s3_parquet(RAW_DATA_BUCKET, "raw_scrapped_data" + file_name_in_dir)
    # Immediately convert `date` column to datetime
    articles['date'] = articles['date'].apply(convert_to_datetime)
    return articles

# --- Import the current modelling dataset --- 
def get_current_modelling_data(ticker):
    file_name_in_dir = f"/{ticker}_modelling_dataset"
    articles = read_s3_parquet(MODELLING_DATA_BUCKET, "modelling_dataset" + file_name_in_dir)
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

    write_log_to_s3(log)

# --- Helper Function: Reading `log.json` file --- 
def read_log_file(): 
    # Therse only one location where our ONLY log file should be held, so no directory is made, rather it's hard coded! 
    log = read_log_from_s3()
    return log


# --- Getting average daily news sentiment ---

def get_params_val(start, end, ticker):
    ticker_data = ticker.history(start=start, end=end, interval="1d").reset_index()
    return float(ticker_data.loc[0,'High']), float(ticker_data.loc[0,'Low']), float(ticker_data.loc[0,'Open']), float(ticker_data.loc[0,'Close'])

# Function for finding daily ticker values; high and low 
def find_params_val(day, yf_ticker):
    """
    find_high_low_daily_val(<datetime>, <yf_ticker>) -> float, float, float, float

    Previously this function only returned the `High` and `Low` daily yf_ticker value. 
    It has been adjusted to also return `Open` and `Close` for further data!
    """
    following_day = day + timedelta(days=1)
    try:
        return get_params_val(day, following_day, yf_ticker)
    except:
        print(f"Error: for {day} and {following_day}")
        return None, None, None, None

# --- Helper Function: Calculating the SMA --- 
def calc_sma(df, target, window):
    return df[target].rolling(window=window).mean()

# Getting SMA 
def get_sma_for_preds(dataframe, window, dropped_cols):
    for target in list(dataframe.drop(columns=dropped_cols).columns):
        col_name_sma = f'{target}_sma'
        dataframe[col_name_sma] = calc_sma(dataframe, target, window)
    return dataframe

def fill_in_na_sma(df):
    # Select rows that have na values 

    null_mask = df.isnull().any(axis=1)
    null_rows = df[null_mask]
    null_rows = null_rows.dropna(axis=1)
    null_rows

    # We'll fill it to be a 5 day moving average for the average per work week (try to)
    null_rows = get_sma_for_preds(null_rows, 5, 'date')

    # Now append this to our final dataset 
    df[null_mask] = null_rows[null_mask]

    df.dropna(inplace=True)

    return df

def format_articles_to_daily(articles, ticker): 
    # Get the aggregated daily sentiment
    daily_sentiment = (
                        articles[['date','sentiment_polarity']]
                       .groupby([articles['date'].dt.floor('D')])
                       .mean('sentiment_polarity')
                       .reset_index()
                       )

    # We need to map our fed in `ticker` value to retrieve the generate `yf.Ticker` value
    map_yf_ticker = {
        "EURUSD=X" : yf.Ticker("EURUSD=X"),
        "JPY=X" : yf.Ticker("JPY=X"),
        "GBPUSD=X" : yf.Ticker("GBPUSD=X"),
    }
    yf_ticker = map_yf_ticker[ticker]

    # Add in daily `High`, `Low`, `Open` & `Close`
    daily_sentiment[['High','Low','Open','Close']] = pd.DataFrame(daily_sentiment['date'].apply(lambda day: find_params_val(day, yf_ticker)).tolist(), index=daily_sentiment.index)

    print("--------------------------------------------")
    print(f"The number of null 'High' and 'Low' values found  \n {daily_sentiment.isnull().sum()}")
    print("--------------------------------------------")
    daily_sentiment.dropna(inplace=True)

    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])

    return daily_sentiment

def save_modelling_data(data, ticker):
    write_s3_parquet(MODELLING_DATA_BUCKET, data, ticker)
    return

def generate_modelling_data(data, ticker):

    tokenizer, finbert = init_finBert()

    # Get the sentiment polarity
    data['sentiment_polarity'] = data['text'].apply(lambda x: analyze_sentiment(tokenizer=tokenizer, finbert=finbert, x=x))
    print("Successfully retrieved the sentiment")

    # Convert the raw formatted data to aggregated daily data 
    data = format_articles_to_daily(data, ticker)
    print("Successfully aggregated the date into daily format")

    # Now add in Simple Moving Averages 
    data = get_sma_for_preds(data.sort_values(by='date',ascending=True), 20, 'date') # Pre-determined window for 20 days; testers/news_articles_modelling.ipynb -- Simple Moving Average

    # There will be dates that have null SMA values because they don't fit in the window; luckily these are in the last 'dates'
    data = fill_in_na_sma(data)

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
    logging_file_exists = file_exists_in_s3(MODELLING_DATA_BUCKET,"modelling_dataset/log.json")
    modelling_data_exists = file_exists_in_s3(MODELLING_DATA_BUCKET, f"modelling_dataset/{ticker}_modelling_dataset")

    print(f"Working on {ticker} raw data!")

    # Get the last date article the modelling data set was updated on, and retrieve work on articles AFTER the last logged article
    try: 
        # If `log.json` even exists..
        if (logging_file_exists and modelling_data_exists):
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
                data = data[~data.duplicated(subset=['date'], keep=False)]
            
            # ******** TO UPDATE ********
            else:
                consolidated_row = pd.DataFrame(columns=data.columns)

            # And then append our consolidated row 
            data = pd.concat([consolidated_row, data], axis = 0).sort_values(by='date', ascending=False)

            # Then save your data
            save_modelling_data(data, ticker)

        elif (logging_file_exists and (not modelling_data_exists)):

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

# - Declare Global Constants to be used all around the program -

# Create a ticker for all currency pairs
EURUSD = "EURUSD=X" # EUR/USD
USDJPY = "JPY=X"
GBPUSD = "GBPUSD=X"

S3_CLIENT = boto3.client('s3')
RAW_DATA_BUCKET = "projectfx608"
MODELLING_DATA_BUCKET = "projectfx608modellingdata"

"""
We have two buckets; 
One to store raw article data as `RAW_DATA_BUCKET`
Another to store modelling data as `MODELLING_DATA_BUCKET
"""

def main():
    print("Initializing ... pipeline: raw data -> modelling data")
    for ticker in [EURUSD, USDJPY, GBPUSD]:
        update_modelling_data(ticker)
    
    # Now run the model_and_predict() line from modelling
    from modelling import model_and_predict
    model_and_predict()

main() # Need this for when intialized manually
