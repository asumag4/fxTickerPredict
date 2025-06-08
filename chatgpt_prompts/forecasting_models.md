Yes, it is indeed possible to use historical data and the patterns derived from it to predict current market data. This type of model is called a **time series forecasting model**. In time series forecasting, we use past values of a variable to predict its future values. 

There are several methods and models you can use for time series forecasting. Some common ones include:

1. **ARIMA (AutoRegressive Integrated Moving Average)**: A classic and widely used time series forecasting method.
2. **LSTM (Long Short-Term Memory)**: A type of recurrent neural network (RNN) that is well-suited for sequence prediction problems.
3. **Prophet**: A tool developed by Facebook for producing high-quality forecasts quickly and easily.

Here is an example of how you might use the **Prophet** library in Python to forecast currency exchange rates:

```python
# Install Prophet
!pip install prophet

import pandas as pd
from prophet import Prophet

# Load historical data
# This is a sample format; replace it with your actual data
data = pd.read_csv('historical_exchange_rates.csv')

# Prepare data for Prophet
data = data.rename(columns={'Date': 'ds', 'ExchangeRate': 'y'})

# Initialize Prophet model
model = Prophet()

# Fit the model
model.fit(data)

# Make future dataframe for predictions
future = model.make_future_dataframe(periods=30)  # Predicting the next 30 days

# Predict future exchange rates
forecast = model.predict(future)

# Display forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# Plot forecast
fig = model.plot(forecast)
fig.show()
```

For the **LSTM** model, you can use the `TensorFlow` or `Keras` library. Here's a simplified example:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and preprocess historical data
data = pd.read_csv('historical_exchange_rates.csv')
data = data[['ExchangeRate']].values

# Scale data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create dataset
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60  # Use past 60 days data to predict the next day
X, y = create_dataset(data_scaled, time_step)

# Reshape data to [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))

# Compile and train model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=50, batch_size=32, verbose=1)

# Predict future exchange rates
future_data = X[-1].reshape((1, time_step, 1))
predicted_value = model.predict(future_data)
predicted_value = scaler.inverse_transform(predicted_value)
print(predicted_value)
```

These examples provide a starting point for building a time series forecasting model in Python. You can experiment with different models and parameters to find the best fit for your specific data and requirements.

Feel free to reach out if you need further guidance or have additional questions!