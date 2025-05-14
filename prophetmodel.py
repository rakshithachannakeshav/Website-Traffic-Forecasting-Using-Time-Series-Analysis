# --- Install necessary packages ---
# Run these in your terminal or Colab cell
# pip install pytrends plotly pandas prophet --upgrade cmdstanpy

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
from sklearn.metrics import mean_absolute_error,root_mean_squared_error


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
model = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.1)
model.fit(df_prophet)

# --- Forecast Future Traffic ---
future = model.make_future_dataframe(periods=12, freq='Y')
forecast = model.predict(future)

# --- Plot Forecast ---
model.plot(forecast)
plt.title('Website Traffic Forecast')
plt.xlabel('Date')
plt.ylabel('Forecasted Rank')
plt.show()

# --- Forecast Components ---
model.plot_components(forecast)
plt.show()

# --- View Forecast Output ---
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))





def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def prophet_forecast(df):
    # Model with tuned parameters
    model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        yearly_seasonality=True
    )
    
    # Add custom seasonality (if applicable)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    model.fit(df)

    # Create future dataframe
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Filter only predictions in original data range
    df_pred = forecast[['ds', 'yhat']].set_index('ds')
    df_true = df.set_index('ds')
    df_combined = df_true.join(df_pred, how='left').dropna()

    print("Combined Data Preview:")
    print(df_combined.head())
    print(f"Rows in combined data: {len(df_combined)}")


    # Metrics
    y_true = df_combined['y']
    y_pred = df_combined['yhat']
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape_val = smape(y_true, y_pred)

    print(f"Prophet RMSE: {rmse:.4f}")
    print(f"Prophet MAE: {mae:.4f}")
    print(f"Prophet MAPE: {mape:.2f}%")
    print(f"Prophet sMAPE: {smape_val:.2f}%")

    # Cross-validation
    df_cv = cross_validation(model, initial='365 days', period='90 days', horizon='90 days')
    df_perf = performance_metrics(df_cv)
    print("\nCross-Validation Performance Metrics:")
    print(df_perf[['horizon', 'rmse', 'mae', 'mape']].head())

    # Plot
    model.plot(forecast)
    plt.title("Prophet Forecast")
    plt.show()

    return forecast
prophet_forecast(df_prophet)
