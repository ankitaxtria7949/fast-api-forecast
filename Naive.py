import numpy as np
import pandas as pd
from dateutil.parser import parse
from datetime import datetime
from dateutil.relativedelta import relativedelta
 
def naive_forecast(data, historyFromDate, historyToDate, selectedFromDate, selectedToDate):
    # Initialize the dictionary with the required format
    forecast_res = {'months': [], 0: []}
    selected_data = {}
    dates = [historyFromDate, historyToDate, selectedFromDate, selectedToDate]
    parsed_dates = []
 
    # Parsing the dates to a more consistent format (MMM-YY)
    for date in dates:
        if date is not None:
            parsed_dates.append(date[4:7] + '-' + date[13:15])
 
    # Standardize date format
    std_dts = []
    for i in data[0]:
        std_dts.append(standardize_date(i))
    data[0] = std_dts
 
    # Get the index of historical start and end dates
    historyFromDateIndex = data[0].index(parsed_dates[0])
    historyToDateIndex = data[0].index(parsed_dates[1])
 
    # Convert the data to a pandas DataFrame and slice based on date indices
    data = pd.DataFrame(data[1:], columns=data[0])
    data = data.iloc[:, historyFromDateIndex:historyToDateIndex + 1]
 
    # Get the historical months between the selected date range
    selected_data["months"] = get_months_between_dates(parsed_dates[2], parsed_dates[3])
    historical_months = data.columns.tolist()
 
    # Initialize a list for forecasted values
    forecast_values = []
 
    for index, row in data.iterrows():
        y = row.values
        y = np.nan_to_num(y, nan=0.0)
 
        # Get the last observed value in the historical data
        last_observed_value = data.iloc[-1].values[-1]  # Last value of the last row
 
        # Repeat the last value for the forecast period
        forecast_values = [last_observed_value] * len(selected_data["months"])
 
        # Set the forecast values and months
        forecast_res['months'] = selected_data["months"]
        forecast_res[0] = list(map(float, forecast_values))# Store the forecasted values
 
        # Add the historical months to the selected_data
        selected_data_1 = {}
        selected_data_1['months'] = historical_months
        selected_data_1[0] = list(map(float, y))
 
    return forecast_res, selected_data_1
 

# Helper function: Standardize Date Format
def standardize_date(date_string):
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
 
 
# Helper function: Get Months Between Dates
def get_months_between_dates(start_date_str, end_date_str):
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
 