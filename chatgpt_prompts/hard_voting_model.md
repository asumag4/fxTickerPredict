To build a hard-voting model that combines multiple forecasting models and predictions from news articles, you can follow these steps:

1. **Train individual models**:
   - Train individual time series forecasting models (ARIMA, LSTM, Prophet) on historical currency data.
   - Train machine learning models (e.g., logistic regression, random forest, support vector machines) on news article data to predict currency movements.

2. **Generate predictions**:
   - Use each trained model to generate predictions for the target currency value.
   - Collect the predictions from all models.

3. **Combine predictions using a hard-voting classifier**:
   - Implement a hard-voting classifier that takes the majority vote of the predictions from the individual models.

Here's an example implementation in Python:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Example function to train ARIMA model and generate predictions
def arima_model(data):
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit()
    prediction = model_fit.forecast(steps=1)
    return prediction[0]

# Example function to train Prophet model and generate predictions
def prophet_model(data):
    df = pd.DataFrame({'ds': data.index, 'y': data.values})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)
    prediction = forecast['yhat'].iloc[-1]
    return prediction

# Example function to train LSTM model and generate predictions
def lstm_model(data):
    data = data.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    X, y = create_dataset(data_scaled, time_step=60)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=50, batch_size=32, verbose=1)
    
    future_data = X[-1].reshape((1, time_step, 1))
    prediction = model.predict(future_data)
    prediction = scaler.inverse_transform(prediction)
    return prediction[0, 0]

# Generate individual model predictions
arima_prediction = arima_model(historical_data)
prophet_prediction = prophet_model(historical_data)
lstm_prediction = lstm_model(historical_data)

# Combine predictions using hard-voting classifier
voting_classifier = VotingClassifier(
    estimators=[
        ('log_reg', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('svc', SVC())
    ],
    voting='hard'
)

# Train and predict with the voting classifier on news article data
X_train_news, y_train_news = ...  # Prepare your news article training data
voting_classifier.fit(X_train_news, y_train_news)

X_test_news = ...  # Prepare your news article test data
news_prediction = voting_classifier.predict(X_test_news)

# Final prediction combining all model predictions
final_prediction = np.round(np.mean([arima_prediction, prophet_prediction, lstm_prediction, news_prediction]))
print(final_prediction)
```

In this example, the predictions from the ARIMA, Prophet, and LSTM models are combined using a hard-voting classifier that also takes into account predictions from news article data. You can experiment with different combinations and parameters to find the best fit for your specific data and requirements.

Feel free to ask if you need further guidance or have additional questions!