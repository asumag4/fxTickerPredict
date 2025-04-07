import streamlit as st

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from modelling import model_and_predict

# Show Raw Data 

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
        name='High Forecasts',
        line=dict(color='red')
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
        name='High',
        line=dict(color='blue')
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

def chart_est_sent(df, ticker):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['sent_pred'],
        mode='lines',
        name='High Prediction',
        line=dict(color='#AB63FA')
    ))

    # Update layout
    fig.update_layout(
        title=f"{ticker} Daily News Sentiment Prediction from FXStreet.com",
        xaxis_title="Date",
        yaxis_title=f"{ticker} Sentiment Polarity",
        legend_title="Legend",
        template="plotly_white"
    )

    return fig

def page_components(ticker):
    # Retrieve the data! 
    pred_data = pd.read_parquet(f"data/prediction_data/{ticker}_predictions_data").sort_values(by='date', ascending=False)
    err_data = pd.read_parquet(f"data/error_data/{ticker}_error_data").sort_values(by='date', ascending=False)
    model_data = pd.read_parquet(f"data/modelling_dataset/{ticker}_modelling_dataset").sort_values(by='date', ascending=False)

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

    # Sentiment 
    st.plotly_chart(chart_est_sent(pred_data, ticker))
    st.markdown("------")

# --- Streamlit Code (main) --- 

st.header("FX-Predictor")

# Init the FX-tickers we are tracking...

# TODO: Might want to turn this into a JSON file that can be accessed by all files to easily intergrate other tickers in one go!
# Create a ticker for all currency pairs

EURUSD = "EURUSD=X" # EUR/USD
USDJPY = "JPY=X"
GBPUSD = "GBPUSD=X"

def render_data():
    tab1, tab2, tab3 = st.tabs([EURUSD, USDJPY, GBPUSD])

    # --- EURUSD info ---
    with tab1:
        page_components(EURUSD)
    with tab2:
        page_components(USDJPY)
    with tab3:
        page_components(GBPUSD)
    
    return tab1, tab2, tab3

# Placeholder for tabs
tabs_placeholder = st.empty()

# Render initial tabs
with tabs_placeholder.container():
    tab1, tab2, tab3 = render_data()

# Spinner to show that data is being processed
with st.spinner(text="Retrieving most recent data!...", show_time=True):
    model_and_predict()  # Backend logic

# Rerender tabs with updated data after `model_and_predict`
with tabs_placeholder.container():
    tab1, tab2, tab3 = render_data()