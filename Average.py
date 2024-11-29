import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error
 
# Define the format mapping for different date formats
format_mapping = {
    "yyyymm": "%Y%m",
    "yyyymmdd": "%Y%m%d",
    "yyyy/mm/dd": "%Y/%m/%d",
    "mm/dd/yyyy": "%m/%d/%Y",
    "dd/mm/yyyy": "%d/%m/%Y",
    "yyyy-mm-dd": "%Y-%m-%d",
    "dd-mm-yyyy": "%d-%m-%Y",
    "mm-dd-yyyy": "%m-%d-%Y",
}
 
# Standardize date function
def standardize_date(date_string):
    try:
        # Check if the date is in MMM-YY format (like Nov-23)
        if len(date_string.split('-')) == 2 and len(date_string.split('-')[1]) == 2 and len(date_string.split('-')[0]) == 3:
            return date_string  # Already in the desired format (MMM-YY)
       
        # Remove time and timezone part from complex date formats
        date_string = re.sub(r'\s[0-9]+:[0-9]+:[0-9]+.*$|GMT[^\)]+', '', date_string)  # Remove time and timezone info
       
        # Try parsing the cleaned-up string using `dateutil.parser`
        parsed_date = parse(date_string, fuzzy=False)
       
        # Return the date in the format "MMM-YY"
        return parsed_date.strftime("%b-%y")  # Output format: Dec-23
    except Exception as e:
        # If parsing fails, return the original date string
        print(f"Error parsing date {date_string}: {e}")
        return date_string
 
# Function to get months between two dates in MMM-YY format
def get_months_between_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%b-%y")
    end_date = datetime.strptime(end_date_str, "%b-%y")
    months_list = []
    current_date = start_date
    while current_date <= end_date:
        months_list.append(current_date.strftime("%b-%y"))
        current_date += relativedelta(months=1)
    return months_list
 
# Function to compute evaluation metrics
def compute_metrics(actual, predicted):
    print("Actual and predicted values",actual,predicted)
    metrics = {}
    # Ensure actual and predicted are numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
   
    # R-squared
    ss_total = np.sum((actual - np.mean(actual))**2)
    ss_residual = np.sum((actual - predicted)**2)
    r_squared = 1 - (ss_residual / ss_total)
    metrics['r_squared'] = r_squared
   
    # Adjusted R-squared
    n = len(actual)  # Number of data points
    p = 1  # Number of predictors (for this simple model, it's just 1)
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    metrics['adj_r_squared'] = adj_r_squared
   
    # RMSE
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    metrics['rmse'] = rmse
   
    # MAE
    mae = mean_absolute_error(actual, predicted)
    metrics['mae'] = mae
   
    # AIC (Akaike Information Criterion)
    n = len(actual)
    residual_sum_of_squares = np.sum((actual - predicted)**2)
    aic = n * np.log(residual_sum_of_squares / n) + 2 * (p + 1)
    metrics['aic'] = aic
   
    # BIC (Bayesian Information Criterion)
    bic = n * np.log(residual_sum_of_squares / n) + np.log(n) * (p + 1)
    metrics['bic'] = bic
   
    # Bias
    bias = np.mean(predicted - actual)
    metrics['bias'] = bias
   
    # Mean Squared Error (MSE)
    mse = mean_squared_error(actual, predicted)
    metrics['mse'] = mse
   
    return metrics
 
# Average method for forecasting
def Average(data, historyFromDate, historyToDate, selectedFromDate, selectedToDate):
    train_data, test_data = split_data(data)
    train_values = train_data[1]
    test_values = test_data[1]
    train_values = list(map(float, train_values))
    test_values = list(map(float, test_values))
    mean_value = np.mean(train_values)
   
    # Predicted values are simply the mean of the training values
    predicted_values = [mean_value] * len(test_values)
   
   
    # Now, extract the actual values from the test data (numeric values)
    actual_values = test_values
    print("Actual and predicted values",actual_values,predicted_values)
    forecast_res = {}
    selected_data = {}
 
    dates = [historyFromDate, historyToDate, selectedFromDate, selectedToDate]
    parsed_dates = []
    for date in dates:
        if date is not None:
            parsed_dates.append(standardize_date(date))
 
    # Standardize the dates in the data
    std_dts = [standardize_date(i) for i in data[0]]
    data[0] = std_dts
 
    historyFromDateIndex = data[0].index(parsed_dates[0])
    historyToDateIndex = data[0].index(parsed_dates[1])
 
    data = pd.DataFrame(data[1:], columns=data[0])
    data = data.iloc[:, historyFromDateIndex:historyToDateIndex + 1]
 
    months = np.array(range(1, len(data.columns) + 1)).reshape(-1, 1)
 
    # Calculate the first and last index for forecasting
    date_format = "%b-%y"
    d1 = datetime.strptime(parsed_dates[0], date_format)
    d2 = datetime.strptime(parsed_dates[2], date_format)
    first_idx = (d2.year - d1.year) * 12 + d2.month - d1.month + 1
 
    d3 = datetime.strptime(parsed_dates[2], date_format)
    d4 = datetime.strptime(parsed_dates[3], date_format)
    last_idx = (d4.year - d3.year) * 12 + d4.month - d3.month + 1
 
    future_months = np.arange(first_idx, first_idx + last_idx).reshape(-1, 1)
 
    start = pd.to_datetime(parsed_dates[2], format='%b-%y')
    end = pd.to_datetime(parsed_dates[3], format='%b-%y')
    date_range = pd.date_range(start=start, end=end, freq='MS')
    col_months = date_range.strftime('%b-%y').tolist()
    forecast_res["months"] = col_months
 
    selected_data["months"] = get_months_between_dates(parsed_dates[0], parsed_dates[1])
    for index, row in data.iterrows():
        forecast = []
        y = row.values.astype(float)
        mean_value = np.mean(y)
        for i in future_months:
            forecast.append(mean_value)
        forecast_res[index] = list(forecast)
        selected_data[index] = list(map(float, y))
    metrics = compute_metrics(actual_values, predicted_values)
 
    return forecast_res, selected_data, metrics
 
# Split data into 80:20 train-test split
def split_data(data, train_size=0.8):
    # Convert the data to a numpy array or just manually split by columns
    split_index = int(len(data[0]) * train_size)
   
    # Split each list (columns of the data) into training and testing parts
    train_data = [col[:split_index] for col in data]
    test_data = [col[split_index:] for col in data]
    print("Test data",test_data)
   
    return train_data, test_data