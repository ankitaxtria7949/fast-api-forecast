def LinearRegression(lassoAlpha, ridgeAlpha, maxiter, modelType, data, historyFromDate, historyToDate, selectedFromDate, selectedToDate):
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    # Initialize the Linear Regression model
    forecast_res = {}
    forecast_for_given_res = {}
    selected_data = {}
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
    data = data.iloc[:, historyFromDateIndex:historyToDateIndex + 1]
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
    forecast_res["months"] = col_months
    selected_data["months"] = get_months_between_dates(parsed_dates[0], parsed_dates[1])
    for index, row in data.iterrows():
        forecast = []
        forecast_for_given = []
        y = row.values.astype(float)  # Convert 'y' values to float
        X = np.hstack((np.ones((months.shape[0], 1)), months))
        if modelType == "Ridge":
            ridge_model = Ridge(alpha=float(ridgeAlpha), fit_intercept=False, max_iter=int(maxiter))
            ridge_model.fit(X, y)
            coefficients = ridge_model.coef_
        elif modelType == "Lasso":
            lasso_model = Lasso(alpha=float(lassoAlpha), fit_intercept=False, max_iter=int(maxiter))
            lasso_model.fit(X, y)
            coefficients = lasso_model.coef_  
        else :
            coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
        temp_X = np.array(range(1, len(X) + 1))
        for i in temp_X:
            forecast_for_given.append(coefficients[0] + coefficients[1] * i)
        for i in future_months:
            forecast.append(coefficients[0] + coefficients[1] * i[0])
        forecast_res[index] = list(forecast)
        forecast_for_given_res[index] = list(forecast_for_given)
        selected_data[index] = list(map(float, y))
    metrics = calculate_metrics(selected_data[0], forecast_for_given_res[0])
    return forecast_res, selected_data, metrics


def calculate_metrics(actual_values, predicted_values):
    import numpy as np
    metrics = {}
    # R-Squared (R²)
    actual_mean = np.mean(actual_values)
    ss_total = np.sum((np.array(actual_values) - actual_mean) ** 2)
    ss_residual = np.sum((np.array(actual_values) - np.array(predicted_values)) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(np.array(actual_values) - np.array(predicted_values)))

    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((np.array(actual_values) - np.array(predicted_values)) / np.array(actual_values))) * 100

    # Mean Squared Error (MSE)
    mse = np.mean((np.array(actual_values) - np.array(predicted_values)) ** 2)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # Normalized Root Mean Squared Error (NRMSE)
    nrmse = rmse / (max(actual_values) - min(actual_values))

    # Weighted Absolute Percentage Error (WAPE)
    wape = np.sum(np.abs(np.array(actual_values) - np.array(predicted_values))) / np.sum(np.abs(actual_values))

    # Weighted Mean Absolute Percentage Error (WMAPE)
    wmape = np.sum(np.abs((np.array(actual_values) - np.array(predicted_values)) / np.array(actual_values)) * np.array(actual_values)) / np.sum(np.abs(actual_values))

    # Store metrics
    metrics["R-Squared"] = r_squared
    metrics["MAE"] = mae
    metrics["MAPE"] = mape
    metrics["MSE"] = mse
    metrics["RMSE"] = rmse
    metrics["NRMSE"] = nrmse
    metrics["WAPE"] = wape
    metrics["WMAPE"] = wmape

    return metrics


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
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
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