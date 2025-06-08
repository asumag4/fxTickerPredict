import streamlit as st
from streamlit_card import card

import pandas as pd
import numpy as np
import plotly.graph_objects as go 

from datetime import datetime

# Show Raw Data 

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

# --- Charting Functions --- 
# --- Charting Function --- 
def chart_est_value(pred_data, model_data, err_data, ticker):
    # Create the plot
    fig = go.Figure()

    # Add high_pred line
    fig.add_trace(go.Scatter(
        x=pred_data['date'],
        y=pred_data['high_pred'],
        mode='lines',
        name='High Forecasts',
        line=dict(color='red')
    ))

    # Add low_pred line
    fig.add_trace(go.Scatter(
        x=pred_data['date'],
        y=pred_data['low_pred'],
        mode='lines',
        name='Low Forecasts',
        line=dict(color='red')
    ))

    # Add filled area between high_pred and low_pred
    fig.add_trace(go.Scatter(
        x=pd.concat([pred_data['date'], pred_data['date'][::-1]]),
        y=pd.concat([pred_data['high_pred'], pred_data['low_pred'][::-1]]),
        fill='toself',
        fillcolor='rgba(100, 0, 0, 0.2)',  # Semi-transparent fill
        line=dict(color='rgba(255,255,255,0)'),  # No visible line
        hoverinfo='skip',
        showlegend=False
    ))

    # -- Add the actual low and high values --

    # Add High
    fig.add_trace(go.Scatter(
        x=model_data['date'],
        y=model_data['High'],
        mode='lines',
        name='High',
        line=dict(color='blue')
    ))

    # Add Low
    fig.add_trace(go.Scatter(
        x=model_data['date'],
        y=model_data['Low'],
        mode='lines',
        name='Low',
        line=dict(color='blue')
    ))

    # Add filled area between high_pred and low_pred
    fig.add_trace(go.Scatter(
        x=pd.concat([model_data['date'], model_data['date'][::-1]]),
        y=pd.concat([model_data['High'], model_data['Low'][::-1]]),
        fill='toself',
        fillcolor='rgba(0, 0, 100, 90)',  # Semi-transparent fill
        line=dict(color='rgba(255,255,255,0)'),  # No visible line
        hoverinfo='skip',
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        title=f"{ticker} Daily High and Low Forecasts",
        xaxis_title="Date",
        yaxis_title=f"{ticker} Exchange Rate",
        legend_title="Legend",
        template="plotly_white"
    )

    # Return the plot 
    return fig

def chart_error_value(err_data, ticker):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=err_data['date'],
        y=err_data['high_pred'],
        name='High Forecasts',
        line=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        x=err_data['date'],
        y=err_data['low_pred'],
        name='Low Forecasts',
        line=dict(color='red')
    ))

    # Add filled area between high_pred and low_pred
    fig.add_trace(go.Scatter(
        x=pd.concat([err_data['date'], err_data['date'][::-1]]),
        y=pd.concat([err_data['high_pred'], err_data['low_pred'][::-1]]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.75)',  # Semi-transparent fill
        line=dict(color='rgba(255,255,255,0)'),  # No visible line
        hoverinfo='skip',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=err_data['date'],
        y=err_data['high_real'],
        name='High',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=err_data['date'],
        y=err_data['low_real'],
        name='Low',
        line=dict(color='blue')
    ))

    # Add filled area between high and low
    fig.add_trace(go.Scatter(
        x=pd.concat([err_data['date'], err_data['date'][::-1]]),
        y=pd.concat([err_data['high_real'], err_data['low_real'][::-1]]),
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.75)',  # Semi-transparent fill
        line=dict(color='rgba(255,255,255,0)'),  # No visible line
        hoverinfo='skip',
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        title=f"{ticker} Error in Forecasts",
        xaxis_title="Date",
        yaxis_title=f"{ticker} Exchange Rate",
        legend_title="Legend",
        template="plotly_white"
    )

    return fig

def calc_rmses(df):
    # Calculate high_rmse
    high_rmse = np.sqrt(((df["high_pred"] - df["high_real"]) ** 2).mean())

    # Calculate low_rmse
    low_rmse = np.sqrt(((df["low_pred"] - df["low_real"]) ** 2).mean())

    # Calculate total_rmse (combined RMSE for high and low predictions)
    tot_rmse = np.sqrt(
        ((df["high_pred"] - df["high_real"]) ** 2).sum() +
        ((df["low_pred"] - df["low_real"]) ** 2).sum()
    ) / len(df)

    return high_rmse, low_rmse, tot_rmse

# --- Recommended Actions Functions ---

def get_ticker_ratio(ticker): # -> return value will be a tuple of (<numerator>, <denominator>)
    # Opted to use if statements, match-case can't access global constants 
    if ticker == EURUSD: 
        return "USD ($)", "EUR (€)"
    if ticker == USDJPY:
        return "JPY (¥)", "USD ($)"
    if ticker == GBPUSD:
        return "USD ($)", "GBP (£)"
    
def generate_action_card(currency, action, ticker):
    if action == BUY:
        return card(
            title=BUY,
            text=currency,
            styles={
                "card": {
                    "width": "90%",
                    "height": "auto",
                    "border-radius": "10%px",
                    "box-shadow": "0 0 10px rgba(0,0,0,0.5)",
                    "background-color": "rgba(0,255,0,1.0)",
                }
            },
            key=f"{currency}_{ticker}"
        )
    if action == SELL:
        return card(
            title=action,
            text=currency,
            styles={
                "card": {
                    "width": "90%",
                    "height": "auto",
                    "border-radius": "10%",
                    "box-shadow": "0 0 10px rgba(0,0,0,0.5)",
                    "background-color": "rgba(255,0,0,1.0)",
                }
            },
            key=f"{currency}_{ticker}"
        )
    if action == HOLD:
        return card(
            title=action,
            text=currency,
            styles={
                "card": {
                    "width": "90%",
                    "height": "auto",
                    "border-radius": "10%",
                    "box-shadow": "0 0 10px rgba(0,0,0,0.5)",
                    "background-color": "rgba(0,0,255,1.0)",
                }
            },
            key=f"{currency}_{ticker}"
        )

# --- Determine which acitons the user should take based on the forecast lol ---

def recommend_actions(ticker, pred_data, err_data): 

    numerator, denominator = get_ticker_ratio(ticker)

    # See if from the most recent recorded high and low values; if the average goes up or down in the next week
    most_recent_high = err_data.iloc[0,-2]
    most_recent_low = err_data.iloc[0,-1]

    # Get the average of the next week projection! 
    today = datetime.today().date()
    next_week_pred = pred_data[pred_data['date'] >= today]
    high_sma_pred_avg = next_week_pred['high_sma_pred'].mean()
    low_sma_pred_avg = next_week_pred['low_sma_pred'].mean()

    
    # LOGIC: 
    # If the ticker value increases; the numerator val decreases, the denominator val increases
    # If the ticker value decreases; the numerator val increases, the denominator val decreases 

    # We'll keep things at a threshold, "increases" is if the next week's forecasted `low_sma` is higher than the 
    # current day's `High`; then tell the user to buy. And vice versa. 

    # We'll have to render the components here unfortunately..
    col4, col5 = st.columns(2)
    with col4:
        # Handling the numerator currency
        if (low_sma_pred_avg > most_recent_high): # If the value for sure increases in total
            # SELL numerator
            generate_action_card(numerator, SELL, ticker)
        elif (high_sma_pred_avg < most_recent_low): # If the value for sure decreases
            # BUY numerator
            generate_action_card(numerator, BUY, ticker)
        else: 
            # The model is unsure 
            generate_action_card(numerator, HOLD, ticker)
    with col5:
        # Handling the denominator currency
        if (low_sma_pred_avg > most_recent_high): # If the value for sure increases in total
            # BUY denominator
            generate_action_card(denominator, BUY, ticker)
        elif (high_sma_pred_avg < most_recent_low): # If the value for sure decreases
            # SELL denominator
            generate_action_card(denominator, SELL, ticker)
        else: 
            # The model is unsure 
            generate_action_card(denominator, HOLD, ticker)

def render_page_components(ticker, desc):
    # Retrieve the data! 
    pred_data = read_s3_parquet(MODELLING_DATA_BUCKET, f"prediction_data/{ticker}_predictions_data_{desc}").sort_values(by='date', ascending=False)
    err_data = read_s3_parquet(MODELLING_DATA_BUCKET, f"error_data/{ticker}_error_data_{desc}").sort_values(by='date', ascending=False)
    model_data = read_s3_parquet(MODELLING_DATA_BUCKET, f"modelling_dataset/{ticker}_modelling_dataset").sort_values(by='date', ascending=False)

    # Show recommneded actions! 
    st.markdown("""#### Recommended Actions""")
    st.text("Based on our 7-day forecast of the trends of the market.")
    recommend_actions(ticker, pred_data, err_data)

    # I want to have a chart like the one rendered by prophet's `plotly_plot()`
    st.plotly_chart(chart_est_value(pred_data, model_data, err_data, ticker))
    st.markdown("------")

    # Then show the error 
    st.plotly_chart(chart_error_value(err_data, ticker))
    st.markdown("------")

    # Show rmse error! 
    st.markdown("""#### Root Mean Squared Error of Historical Forecasts""")
    high_rmse, low_rmse, tot_rmse = calc_rmses(err_data)
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE of 'High'", np.round(high_rmse, 5))
    col2.metric("RMSE of 'Low'", np.round(low_rmse, 5))
    col3.metric("Total RMSE", np.round(tot_rmse, 5))
    st.markdown("------")

# --- Streamlit Code (main) --- 

st.header("FX-Predictor")

# Init the FX-tickers we are tracking...

# TODO: Might want to turn this into a JSON file that can be accessed by all files to easily intergrate other tickers in one go!
# Create a ticker for all currency pairs

S3_CLIENT = boto3.client('s3')
MODELLING_DATA_BUCKET = "projectfx608modellingdata"

EURUSD = "EURUSD=X" # EUR/USD
USDJPY = "JPY=X"
GBPUSD = "GBPUSD=X"

BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"

def render_data():
    tab1, tab2, tab3 = st.tabs([EURUSD, USDJPY, GBPUSD])

    # --- EURUSD info ---
    with tab1:
        tab1_check = st.checkbox("Exclude News Sentiment Predictors", key="tab1_checkbox")
        if tab1_check:
            render_page_components(EURUSD, "no_sent")
        else: 
            render_page_components(EURUSD, "complete")
    with tab2:
        tab2_check = st.checkbox("Exclude News Sentiment Predictors", key="tab2_checkbox")
        if tab2_check:
            render_page_components(USDJPY, "no_sent")
        else: 
            render_page_components(USDJPY, "complete")
    with tab3:
        tab3_check = st.checkbox("Exclude News Sentiment Predictors", key="tab3_checkbox")
        if tab3_check:
            render_page_components(GBPUSD, "no_sent")
        else: 
            render_page_components(GBPUSD, "complete")
    
    return tab1, tab2, tab3

# Placeholder for tabs
tabs_placeholder = st.empty()

# Render initial tabs
with tabs_placeholder.container():
    tab1, tab2, tab3 = render_data()

