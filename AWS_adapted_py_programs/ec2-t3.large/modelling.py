import yfinance as yf

import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
from pathlib import Path

from format_to_model import main as format_to_model

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

def write_s3_parquet(Bucket, data, Key, ticker):
    # This function is to write a the modelling_dataset to parquet dataframe to S3 as a parquet file

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

def write_err_data_to_s3_parquet(Bucket, data, Key, ticker):
    # This function is to write a the modelling_dataset to parquet dataframe to S3 as a parquet file

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

# --- Helper Function: Function to check if a file exists in S3 ---
def file_exists_in_s3(bucket_name, s3_key):
    try:
        S3_CLIENT.head_object(Bucket=bucket_name, Key=s3_key)
        return True
    except Exception as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise

# ------ Regular Functions Adopted to AWS ------

# --- Helper: Determines the predictions of the data --- 
def record_predictions(current_pred, ticker, desc):
    pred_data_key = f"prediction_data/{ticker}_predictions_data_{desc}"
    prediction_data_exists = file_exists_in_s3(MODELLING_DATA_BUCKET, pred_data_key)

    if prediction_data_exists:
        print(f"Record of predictions for {ticker} was found!")
        past_pred = read_s3_parquet(MODELLING_DATA_BUCKET, pred_data_key)
    else:
        print(f"No record of predictions for {ticker} was found... initializing file data to be written!")
        past_pred = pd.DataFrame(columns=['date', 'high_pred', 'low_pred', 'sent_pred'])

    # If there are rows with duplicate `date` (lets say we're getting new current data and the model is refit for the day)
    filtered_past_pred = past_pred[~past_pred['date'].isin(current_pred['date'])]
    to_write = pd.concat([current_pred, filtered_past_pred])
    
    write_s3_parquet(MODELLING_DATA_BUCKET, to_write, pred_data_key, ticker)
    return

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

    data = read_s3_parquet(MODELLING_DATA_BUCKET, f"modelling_dataset/{ticker}_modelling_dataset")

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

    file_path = f"error_data/{ticker}_error_data_{desc}"
    err_data_exists = file_exists_in_s3(MODELLING_DATA_BUCKET, file_path)

    if err_data_exists:
        print(f"Record of error predictions for {ticker} was found!")
        past_err = read_s3_parquet(MODELLING_DATA_BUCKET, file_path)
    else:
        print(f"No record of predictions for {ticker} was found... initializing file data to be written!")
        past_err = pd.DataFrame(columns=['date', 'high_pred', 'low_pred', 'high_real', 'low_real'])

    # If there are rows with duplicate `date` (lets say we're getting new current data and the model is refit for the day) 
    filtered_past_err = past_err[~past_err['date'].isin(now_err['date'])]
    updated_err = pd.concat([now_err, filtered_past_err])

    updated_err = updated_err.dropna()

    # Save this data! 
    write_err_data_to_s3_parquet(MODELLING_DATA_BUCKET, updated_err, file_path, ticker)



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

S3_CLIENT = boto3.client('s3')
RAW_DATA_BUCKET = "projectfx608"
MODELLING_DATA_BUCKET = "projectfx608modellingdata"

def model_and_predict():
    # format_to_model()
    print("Initializing ... pipeline: modeling data -> predicted & error data")
    for ticker in [EURUSD, USDJPY, GBPUSD,]: 
        generate_predictions(ticker)
        
model_and_predict() # For when initialized independently
