import yfinance as yf

import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
from pathlib import Path

from formatting_to_model import main as format_to_model

# --- Helper: Determines the predictions of the data --- 
def record_predictions(current_pred, ticker):
    file_path = f"data/prediction_data/{ticker}_predictions_data"
    logging_file = Path(file_path)

    if logging_file.is_file():
        print(f"Record of predictions for {ticker} was found!")
        past_pred = pd.read_parquet(file_path)
    else:
        print(f"No record of predictions for {ticker} was found... initializing file data to be written!")
        past_pred = pd.DataFrame(columns=['date', 'high_pred', 'low_pred', 'sent_pred'])

    # If there are rows with duplicate `date` (lets say we're getting new current data and the model is refit for the day)
    filtered_past_pred = past_pred[~past_pred['date'].isin(current_pred['date'])]
    to_write = pd.concat([current_pred, filtered_past_pred])
    
    to_write.to_parquet(f"data/prediction_data/{ticker}_predictions_data")
    return

# --- Function to generate sentiment polarity predictions to be fed to final model ---
def generate_sentiment_predictions(data):
    # Get the sentiment data
    sentiment_df = data[['date','sentiment_polarity']].rename(mapper = {"date" : "ds", "sentiment_polarity" : "y"}, axis = 1)

    # Init and train a model to it
    m_sent = Prophet() # m_sent = model sentiment
    m_sent.fit(sentiment_df)

    # Predict for the next 7 days
    f_sent = pd.DataFrame({"ds" : [datetime.now().date() + timedelta(days= i) for i in range(0,8)]}) # f_sent = future sentiment
    f_sent.head()

    fc_sent = m_sent.predict(f_sent)

    return fc_sent[['ds','yhat']]

# --- Generates Prophet Model to predict ticker values with Sentiment ---
def prophet_with_sentiment(data, target, sent_regr):

    ticker_df = data[['date', target,'sentiment_polarity']].rename(mapper = {"date" : "ds", target : "y"}, axis = 1)

    m_ticker = Prophet()
    m_ticker.add_regressor('sentiment_polarity')
    m_ticker.fit(ticker_df)

    # Predict for the next 7 days! 
    future = pd.DataFrame({"ds" : [datetime.now() + timedelta(days= i) for i in range(0,8)], "sentiment_polarity" : sent_regr['yhat']})

    forecast = m_ticker.predict(future)

    return forecast[['ds','yhat']]

# --- Function to consolidate high and low data --- 

def generate_predictions(ticker):
    data = pd.read_parquet(f"data/modelling_dataset/{ticker}_modelling_dataset")

    sent_regr = generate_sentiment_predictions(data)

    high_df = prophet_with_sentiment(data, "High", sent_regr).rename(mapper = {"ds" : "date", "yhat" : "high_pred"}, axis = 1)
    low_df = prophet_with_sentiment(data, "Low", sent_regr).rename(mapper = {"ds" : "date", "yhat" : "low_pred"}, axis = 1)
    
    # Theres an error in that the slight delay in seconds execution in predicting High and Low data will cause an issue, fix that:
    sent_regr = sent_regr.rename(mapper = {"ds" : "date", "yhat" : "sent_pred"}, axis = 1)
    sent_regr['date'] = sent_regr['date'].dt.date
    high_df['date'] = high_df['date'].dt.date
    low_df['date'] = low_df['date'].dt.date

    predicts_merged = pd.merge(high_df, low_df, on='date')
    predicts_merged = pd.merge(predicts_merged, sent_regr, on='date')
    print("DEBUG: see data merged!")
    print(predicts_merged.info())

    record_predictions(predicts_merged, ticker)

    return predicts_merged

# --- Helper Function: Getting the published date! ---
def get_published_ticker_vals(day, ticker, target):
    ticker = yf.Ticker(ticker)
    day_after = day + timedelta(days=1)
    real_data = ticker.history(start=day, end=day_after, interval="1d").reset_index()
    if real_data.empty: # Is empty
        return None
    else: 
        return real_data[target].iloc[0]

# --- Generate Errors if possible --- 
def generate_errors_ticker(now_err, ticker):

    # In the current predictions, the current day will give updated highs and lows! 
    for target in ['High', 'Low']:
        col_target_name = ERR_COL_MAPPINGS[target]
        now_err[col_target_name] = now_err['date'].apply(lambda x: get_published_ticker_vals(x, ticker, target))

    file_path = f"data/error_data/{ticker}_error_data"
    error_data = Path(file_path)

    if error_data.is_file():
        print(f"Record of error predictions for {ticker} was found!")
        past_err = pd.read_parquet(file_path)
    else:
        print(f"No record of predictions for {ticker} was found... initializing file data to be written!")
        past_err = pd.DataFrame(columns=['date', 'high_pred', 'low_pred', 'high_real', 'low_real'])

    # If there are rows with duplicate `date` (lets say we're getting new current data and the model is refit for the day) 
    filtered_past_err = past_err[~past_err['date'].isin(now_err['date'])]
    updated_err = pd.concat([now_err, filtered_past_err])

    updated_err = updated_err.dropna()

    # Save this data! 
    updated_err.to_parquet(f"data/error_data/{ticker}_error_data")

    return updated_err 
    # We'll calculate error later!

# --- RUNNING THE PROGRAM --- 

# Global Vars
# Create a ticker for all currency pairs
EURUSD = "EURUSD=X" # EUR/USD
USDJPY = "JPY=X"
GBPUSD = "GBPUSD=X"

# Map for error predictions 
ERR_COL_MAPPINGS = {
    'High' : 'high_real',
    'Low' : 'low_real',
}

def model_and_predict():
    format_to_model()
    for ticker in [EURUSD, USDJPY, GBPUSD]:
        new_preds = generate_predictions(ticker)
        generate_errors_ticker(new_preds, ticker)

model_and_predict()
