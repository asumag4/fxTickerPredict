import yfinance as yf

import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
from pathlib import Path

from format_to_model import main as format_to_model

# --- Helper: Determines the predictions of the data --- 
def record_predictions(current_pred, ticker, desc):
    file_path = f"data/prediction_data/{ticker}_predictions_data_{desc}"
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
    
    to_write.to_parquet(f"data/prediction_data/{ticker}_predictions_data_{desc}")

"""
# ---  DEPRECATED (ONLY FOR REFERENCE): Function to generate sentiment polarity predictions to be fed to final model ---
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

# --- DEPRECATED (ONLY FOR REFERENCE): Generates Prophet Model to predict ticker values with Sentiment ---
def prophet_with_sentiment(data, target, sent_regr):

    ticker_df = data[['date', target,'sentiment_polarity']].rename(mapper = {"date" : "ds", target : "y"}, axis = 1)

    m_ticker = Prophet()
    m_ticker.add_regressor('sentiment_polarity')
    m_ticker.fit(ticker_df)

    # Predict for the next 7 days! 
    future = pd.DataFrame({"ds" : [datetime.now() + timedelta(days= i) for i in range(0,8)], "sentiment_polarity" : sent_regr['yhat']})

    forecast = m_ticker.predict(future)

    return forecast[['ds','yhat']]
"""

# --- Modelling Function --- 
def generate_future_predictors_prophet(data, range_end): 
    forecasts = {}
    for predictor in data.drop(columns=['date','High','Low']).columns:
        # Prep the data to fit prophet's requirements
        df_fit = data[['date',predictor]].rename(mapper = {"date" : "ds", predictor : "y"}, axis = 1)

        # Init the model 
        m = Prophet()
        m.fit(df_fit)

        # 1. Generate future df to store forecast
        future = pd.DataFrame({"ds" : [datetime.now() + timedelta(days= i) for i in range(0,range_end)]})
        # 2. Forecast
        forecast = m.predict(future)

        if 'ds' not in list(forecasts.keys()): 
            forecasts['ds'] = forecast['ds']
        
        forecasts[(predictor.lower() + "_pred")] = forecast['yhat']



    forecasts = pd.DataFrame(forecasts)
    return forecasts

def model_and_predict_prophet(data, target, pred_regr):

    if target == "High":
        opp_target = "Low"
    else:
        opp_target = "High"

    data = data.drop(columns=opp_target) # The data still needs the target though; for fitting how target moves with predictors

    # Forecasting the target variable
    train_df = data.rename(mapper = {"date" : "ds", target : "y"}, axis = 1)

    # We're gonna have to rename our `_pred` columns back to their original in order for the model to work with out predicted data...
    rename_map = dict(zip(list(pred_regr.drop(columns='ds').columns), list(data.drop(columns=['date',target]).columns)))

    m = Prophet()

    # We now need to add all the regressors 
    for predictor in list(data.drop(columns=['date',target]).columns):
        m.add_regressor(predictor)

    m.fit(train_df)

    # Since we're testing with current data: 
    future = pd.DataFrame({"ds" : [datetime.now() + timedelta(days= i) for i in range(0,8)]})
    future = pd.concat([future, pred_regr.drop(columns='ds').rename(columns=rename_map)], axis=1)

    forecast = m.predict(future)
    return forecast[['ds','yhat']]


# --- Function to consolidate high and low data --- 

def generate_predictions(ticker):

    data = pd.read_parquet(f"data/modelling_dataset/{ticker}_modelling_dataset")
    
    for drop_columns in [[],['sentiment_polarity','sentiment_polarity_sma',]]: 

        data1 = data.drop(columns=drop_columns)

        # Generate the future predictors first 
        pred_regr = generate_future_predictors_prophet(data1, 8) # Hard coded to 7 days (1 week) forecast

        high_df = model_and_predict_prophet(data1, "High", pred_regr).rename(mapper = {"ds" : "date", "yhat" : "high_pred"}, axis = 1)
        low_df = model_and_predict_prophet(data1, "Low", pred_regr).rename(mapper = {"ds" : "date", "yhat" : "low_pred"}, axis = 1)
        
        # Theres an error in that the slight delay in seconds execution in predicting High and Low data will cause an issue, fix that:
        pred_regr = pred_regr.rename(mapper = {"ds" : "date"}, axis = 1)
        pred_regr['date'] = pred_regr['date'].dt.date
        high_df['date'] = high_df['date'].dt.date
        low_df['date'] = low_df['date'].dt.date

        predicts_merged = pd.merge(high_df, low_df, on='date')
        predicts_merged = pd.merge(predicts_merged, pred_regr, on='date')
        print("DEBUG: see data merged!")
        print(predicts_merged.info())

        if drop_columns:
            desc = "no_sent"
            record_predictions(predicts_merged, ticker, desc)
            generate_errors_ticker(predicts_merged, ticker, desc)
        else:
            desc = "complete"
            record_predictions(predicts_merged, ticker, desc)
            generate_errors_ticker(predicts_merged, ticker, desc)


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
def generate_errors_ticker(now_err, ticker, desc):

    # In the current predictions, the current day will give updated highs and lows! 
    for target in ['High', 'Low']:
        col_target_name = ERR_COL_MAPPINGS[target]
        now_err[col_target_name] = now_err['date'].apply(lambda x: get_published_ticker_vals(x, ticker, target))

    file_path = f"data/error_data/{ticker}_error_data_{desc}"
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
    updated_err.to_parquet(f"data/error_data/{ticker}_error_data_{desc}")

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
    # format_to_model()
    print("Initializing ... pipeline: modeling data -> predicted & error data")
    for ticker in [EURUSD, USDJPY, GBPUSD,]: 
        generate_predictions(ticker)
        

model_and_predict() # For when initialized independently
