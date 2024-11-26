from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd

def LinearRegression(data, historyFromDate, historyToDate, selectedFromDate, selectedToDate):
    # Initialize the Linear Regression model
    forecast_res = {}
    selected_data ={}
    dates = [historyFromDate, historyToDate, selectedFromDate, selectedToDate]
    parsed_dates = []
    for date in dates:
        if date is not None:
            parsed_dates.append(date[4:7] + '-' + date[13:15])

    std_dts = []
    for i in data[0]:
        std_dts.append(standardize_date(i))
    data[0] = std_dts
    historyFromDateIndex = data[0].index(parsed_dates[0])
    historyToDateIndex = data[0].index(parsed_dates[1])
    data = pd.DataFrame(data[1:], columns=data[0])
    data = data.iloc[:, historyFromDateIndex:historyToDateIndex+1]
    months = np.array(range(1, len(data.columns) + 1)).reshape(-1, 1)

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
    forecast_res["months"]=col_months
    selected_data["months"] = get_months_between_dates(parsed_dates[0], parsed_dates[1])
    for index, row in data.iterrows():
        y = row.values
        print(y)
        print(months)
        model = LinearRegressionScratch()
        model.fit(months, y, epochs=1000, learning_rate=0.01)
        forecast = model.predict(future_months)
        forecast_res[index] = list(forecast)
        selected_data[index] = list(map(float, y))
    return forecast_res, selected_data


class LinearRegressionScratch:
    def __init__(self):
        self.weights = None  # Coefficients
        self.bias = None     # Intercept

    def fit(self, X, y, epochs=1000, learning_rate=0.01):
        """Train the linear regression model using gradient descent."""
        n_samples = len(X)  # Number of data points
        n_features = len(X[0])  # Number of features

        # Initialize weights and bias
        self.weights = [0] * n_features
        self.bias = 0

        for _ in range(epochs):
            # Predicted values
            y_pred = [self.predict_single(x) for x in X]

            # Compute gradients
            weight_gradients = [
                -2 / n_samples * sum((y[i] - y_pred[i]) * X[i][j] for i in range(n_samples))
                for j in range(n_features)
            ]
            bias_gradient = -2 / n_samples * sum(y[i] - y_pred[i] for i in range(n_samples))

            # Update weights and bias
            self.weights = [w - learning_rate * g for w, g in zip(self.weights, weight_gradients)]
            self.bias -= learning_rate * bias_gradient

    def predict(self, X):
        """Predict using the linear model."""
        return [self.predict_single(x) for x in X]

    def predict_single(self, x):
        """Predict a single instance."""
        return sum(w * x_j for w, x_j in zip(self.weights, x)) + self.bias


def standardize_date(date_string):
    from dateutil.parser import parse
    try:
        # Check if the date is in MMM-YY format (like Nov-23)
        if len(date_string.split('-')) == 2 and len(date_string.split('-')[1]) == 2 and len(date_string.split('-')[0]) == 3:
            return date_string
        else:
            # If it's a full date, use the standard date parsing
            parsed_date = parse(date_string)
            return parsed_date.strftime("%b-%y")  # Output format: Nov-23
    except:
        return date_string





def get_months_between_dates(start_date_str, end_date_str):
    # Define the format of the input dates
    date_format = "%b-%y"
    
    # Parse the start and end dates
    start_date = datetime.strptime(start_date_str, date_format)
    end_date = datetime.strptime(end_date_str, date_format)
    
    # List to store the months between the dates
    months_list = []
    
    # Loop through the months between the start and end date
    current_date = start_date
    while current_date <= end_date:
        months_list.append(current_date.strftime(date_format))  # Add the current month to the list
        current_date += relativedelta(months=1)  # Move to the next month
    
    return months_list