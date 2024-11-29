def Holt(data, historyFromDate, historyToDate, selectedFromDate, selectedToDate):
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    # Initialize the Linear Regression model
    forecast_res = {}
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



    start = pd.to_datetime(parsed_dates[2], format='%b-%y')
    end = pd.to_datetime(parsed_dates[3], format='%b-%y')


    date_range = pd.date_range(start=start, end=end, freq='MS')
    col_months = date_range.strftime('%b-%y').tolist()
    forecast_res["months"] = col_months
    selected_data["months"] = get_months_between_dates(parsed_dates[0], parsed_dates[1])


    forecast = []
    metrics = {}
    for index, row in data.iterrows():
        y = row.values.astype(float)
        selected_data[index] = list(map(float, y))
        y = list(y)

        split_index = int(len(list(y)) * 0.7)
        part1 = y[:split_index]
        part2 = y[split_index:]

        model = ExponentialSmoothing(
            pd.Series(y),
            trend="additive",        # Captures the trend in the data
            seasonal="additive",     # Assumes additive seasonal effects
            seasonal_periods=6,     # Yearly seasonality (12 months)
            initialization_method="estimated"
        )
        fit = model.fit()
        forecast_steps = len(get_months_between_dates(parsed_dates[1], parsed_dates[3]))-1
        forecast = fit.forecast(steps=forecast_steps)
        forecast = list(forecast)
        to_remove = len(get_months_between_dates(parsed_dates[1], parsed_dates[2]))-2
        forecast = forecast[to_remove:]
        forecast_res[index] = list(forecast)
        
        model_eval = ExponentialSmoothing(
        pd.Series(part1), trend="additive", seasonal=None, initialization_method="estimated"
        )
        model_eval = model_eval.fit()
        forecast = model_eval.forecast(steps=len(part2))
        forecast = list(forecast)
        metrics = calculate_metrics(part2, forecast)
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