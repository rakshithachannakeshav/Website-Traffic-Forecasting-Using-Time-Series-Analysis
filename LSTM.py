# --- Install necessary packages ---
# Run these in your terminal or Colab cell
# pip install pytrends plotly pandas prophet tensorflow scikit-learn --upgrade cmdstanpy

# --- Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pytrends.request import TrendReq
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Set plot renderer to browser
pio.renderers.default = 'browser'

# --- Load and Clean Dataset ---
df = pd.read_csv(r"C:\Users\Meghana\OneDrive\Desktop\meghna\.vs studio\websiteTraffic\trends.csv.zip")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df['year'] = df['year'].astype(int)

# --- Top 15 Brands ---
top_brands = df['query'].value_counts().head(15).index
df_top = df[df['query'].isin(top_brands)]

# --- Pivot Table for Trend Line ---
df_pivot = df_top.pivot_table(index='year', columns='query', values='rank', aggfunc='mean')
df_pivot = df_pivot.ffill()

# --- Line Plot: Rank Over Time ---
plt.figure(figsize=(14, 7))
for brand in df_pivot.columns:
    plt.plot(df_pivot.index, df_pivot[brand], label=brand)
plt.gca().invert_yaxis()
plt.title('Rank Over Time (Top 15 Brands)')
plt.xlabel('Year')
plt.ylabel('Rank')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Bar Chart: Top Brands by Frequency ---
plt.figure(figsize=(8, 6))
df['query'].value_counts().head(15).plot(kind='barh', color='teal')
plt.title('Top 15 Brands by Frequency')
plt.xlabel('Appearances')
plt.tight_layout()
plt.show()

# --- Heatmap: Average Rank by Year ---
heatmap_data = df_top.pivot_table(index='query', columns='year', values='rank', aggfunc='mean')
plt.figure(figsize=(12, 7))
sns.heatmap(heatmap_data, cmap='mako', linewidths=0.5)
plt.title('Heatmap: Average Rank of Top 15 Brands Over Years')
plt.xlabel('Year')
plt.ylabel('Brand')
plt.tight_layout()
plt.show()

# --- Boxplot: Rank Distribution by Brand ---
plt.figure(figsize=(14, 6))
sns.boxplot(data=df_top, x='query', y='rank')
plt.title('Rank Distribution by Brand (Top 15)')
plt.xticks(rotation=45)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# --- Average Rank of All Brands Over Time ---
avg_rank_all = df.groupby('year')['rank'].mean()
plt.figure(figsize=(10, 5))
plt.plot(avg_rank_all.index, avg_rank_all.values, marker='o', color='purple')
plt.title('Average Rank of All Brands Over Time')
plt.xlabel('Year')
plt.ylabel('Average Rank')
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Unique Brands Per Year ---
brands_per_year = df.groupby('year')['query'].nunique()
plt.figure(figsize=(10, 5))
plt.bar(brands_per_year.index, brands_per_year.values, color='coral')
plt.title('Number of Unique Brands Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Brands')
plt.tight_layout()
plt.show()

# --- Interactive Line Plot: Top Brands ---
fig = px.line(df_top, x='year', y='rank', color='query',
              title='Interactive Line Plot - Top 15 Brands')
fig.update_yaxes(autorange='reversed')
fig.update_layout(height=600)
fig.show()

# --- Interactive Pie Chart ---
brand_counts = df['query'].value_counts().head(15).reset_index()
brand_counts.columns = ['Brand', 'Frequency']
fig_pie = px.pie(brand_counts, names='Brand', values='Frequency',
                 title='Top 15 Brands - Frequency Distribution',
                 hole=0.3)
fig_pie.update_traces(textinfo='percent+label')
fig_pie.show()

# --- Prepare Data for Prophet ---
df_prophet = df[['year', 'rank']].copy()
df_prophet.rename(columns={'year': 'ds', 'rank': 'y'}, inplace=True)
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y')
df_prophet['y'] = df_prophet['y'].astype(float)

# --- Prophet Model ---
model_prophet = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.1)
model_prophet.fit(df_prophet)

# --- Forecast Future Traffic with Prophet ---
future = model_prophet.make_future_dataframe(periods=12, freq='Y')
forecast = model_prophet.predict(future)

# --- Plot Forecast Prophet ---
model_prophet.plot(forecast)
plt.title('Website Traffic Forecast with Prophet')
plt.xlabel('Date')
plt.ylabel('Forecasted Rank')
plt.show()

# --- Forecast Components Prophet ---
model_prophet.plot_components(forecast)
plt.show()

# --- View Forecast Output Prophet ---
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))

# --- Cross-validation and Evaluation (Prophet) ---
df_cv = cross_validation(model_prophet, initial='730 days', period='180 days', horizon='365 days')
df_p = performance_metrics(df_cv)
print(df_p[['horizon', 'rmse', 'mape']].head())

# --- Prepare Data for LSTM ---
df_lstm = df[['year', 'rank']].copy()
df_lstm['year'] = pd.to_datetime(df_lstm['year'], format='%Y')
df_lstm.set_index('year', inplace=True)

# --- Scaling the data for LSTM ---
scaler = MinMaxScaler(feature_range=(0, 1))
df_lstm_scaled = scaler.fit_transform(df_lstm)

# --- Creating Sequences for LSTM ---
def create_lstm_sequences(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 5
X_lstm, y_lstm = create_lstm_sequences(df_lstm_scaled, time_step)

# --- Reshape X_lstm for LSTM input ---
X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

# --- Train-Test Split for LSTM ---
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, shuffle=False)

# --- LSTM Model ---
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(units=50, return_sequences=False))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(units=1))

# --- Compile and Fit LSTM Model ---
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train, y_train, epochs=50, batch_size=32)

# --- Make Predictions with LSTM ---
predictions_lstm = model_lstm.predict(X_test)
predictions_lstm = scaler.inverse_transform(predictions_lstm)

# --- Plot LSTM Forecast vs Actual ---
plt.figure(figsize=(10, 5))
plt.plot(df_lstm.index[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), color='blue', label='Actual Rank')
plt.plot(df_lstm.index[-len(predictions_lstm):], predictions_lstm, color='red', label='LSTM Predicted Rank')
plt.title('LSTM Model: Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Rank')
plt.legend()
plt.show()


# --- LSTM Model Evaluation ---
mse_lstm = mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), predictions_lstm)
mae_lstm = mean_absolute_error(scaler.inverse_transform(y_test.reshape(-1, 1)), predictions_lstm)

# MAPE calculation
y_true_lstm = scaler.inverse_transform(y_test.reshape(-1, 1))
mape_lstm = np.mean(np.abs((y_true_lstm - predictions_lstm) / y_true_lstm)) * 100

print(f'LSTM Mean Squared Error: {mse_lstm}')
print(f'LSTM Mean Absolute Error: {mae_lstm}')
print(f'LSTM Mean Absolute Percentage Error (MAPE): {mape_lstm:.2f}%')


# --- Forecast Future Years with LSTM ---
def predict_future_lstm(model, last_data, future_steps, time_step):
    predictions = []
    current_input = last_data

    # Predicting the future values
    for _ in range(future_steps):
        # Reshape current_input for LSTM prediction
        current_input = current_input.reshape(1, time_step, 1)
        prediction = model.predict(current_input)
        
        # Append the predicted value to the predictions list
        predictions.append(prediction[0, 0])
        
        # Update current_input with the new prediction (shift the window)
        current_input = np.append(current_input[0][1:], prediction).reshape(time_step, 1)

    return predictions


# --- Get the last time_step data from the training set ---
last_data = df_lstm_scaled[-time_step:]

# --- Forecast for next 'n' years (for example, next 5 years) ---
future_steps = 5  # Number of future years to forecast
predictions_lstm_future = predict_future_lstm(model_lstm, last_data, future_steps, time_step)

# --- Inverse transform predictions to original scale ---
predictions_lstm_future = scaler.inverse_transform(np.array(predictions_lstm_future).reshape(-1, 1))

# --- Print Forecasted Future Values ---
future_years = pd.date_range(df_lstm.index[-1] + pd.DateOffset(years=1), periods=future_steps, freq='Y')
forecast_df = pd.DataFrame(data=predictions_lstm_future, index=future_years, columns=['Forecasted Rank'])

print(forecast_df)


# --- Plot Future Forecasted Values ---
plt.figure(figsize=(10, 5))
plt.plot(df_lstm.index[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), color='blue', label='Actual Rank')
plt.plot(future_years, predictions_lstm_future, color='red', label='LSTM Forecasted Rank')
plt.title('LSTM Forecast vs Actual (Future Predictions)')
plt.xlabel('Year')
plt.ylabel('Rank')
plt.legend()
plt.show()
