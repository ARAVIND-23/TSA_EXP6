# Ex.No: 6               HOLT WINTERS METHOD
### Date: 01.9.26
### Reg No:212223240011
### Name: ARAVIND G

### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:

```

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data and inspect columns
df = pd.read_csv('/content/bmw (1).csv')
print("Columns in the dataset:", df.columns.tolist())

# Prepare time series data: mean price per year
df['year_dt'] = pd.to_datetime(df['year'], format='%Y') # Convert 'year' to datetime objects
df = df.set_index('year_dt').sort_index() # Set 'year_dt' as index and sort
bmw_price_ts = df.resample('YS')['price'].mean().dropna() # Resample to Year Start (YS) and take mean price

plt.figure(figsize=(12, 6))
bmw_price_ts.plot(title='BMW Mean Price by Year')
plt.xlabel('Year')
plt.ylabel('Mean Price')
plt.show()

# Split test,train data for annual time series
train_size_annual = int(len(bmw_price_ts) * 0.8)
train_data_annual = bmw_price_ts[:train_size_annual]
test_data_annual = bmw_price_ts[train_size_annual:]

# Create and train Holt-Winters model on annual data (trend='add', no seasonality for yearly data)
model_annual = ExponentialSmoothing(train_data_annual, trend='add', seasonal=None).fit()
test_predictions_annual = model_annual.forecast(steps=len(test_data_annual))

# Plot actual vs. predicted for test set
plt.figure(figsize=(12, 6))
ax = train_data_annual.plot(label='Train Data')
test_data_annual.plot(ax=ax, label='Actual Test Data')
test_predictions_annual.plot(ax=ax, label='Predicted Test Data')
ax.legend()
ax.set_title('Annual BMW Price: Train, Test, and Predictions')
plt.xlabel('Year')
plt.ylabel('Mean Price')
plt.show()

# Evaluate the model
rmse_annual = np.sqrt(mean_squared_error(test_data_annual, test_predictions_annual))
print(f"RMSE for annual predictions: {rmse_annual:.2f}")

# Create final model with all data and forecast future years
final_model_annual = ExponentialSmoothing(bmw_price_ts, trend='add', seasonal=None).fit()
future_forecast_steps_annual = 5 # Forecast for the next 5 years
future_predictions_annual = final_model_annual.forecast(steps=future_forecast_steps_annual)

# Plot historical data and future predictions
plt.figure(figsize=(12, 6))
ax = bmw_price_ts.plot(label='Historical Data')
future_predictions_annual.plot(ax=ax, label='Future Forecast')
ax.legend()
ax.set_title('Annual BMW Price: Historical Data and Future Forecast')
plt.xlabel('Year')
plt.ylabel('Mean Price')
plt.show()

```


### OUTPUT:

<img width="912" height="452" alt="image" src="https://github.com/user-attachments/assets/d39c0042-7008-4e70-a6a1-8212859b9ce1" />


TEST_PREDICTION:

<img width="867" height="451" alt="Screenshot 2026-08-28 113920" src="https://github.com/user-attachments/assets/76ebffb9-3120-4e2f-b9e3-053754799119" />


FINAL_PREDICTION:
<img width="867" height="473" alt="Screenshot 2026-08-28 113254" src="https://github.com/user-attachments/assets/45de65cc-1a88-4d30-a379-5131e2dae956" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
