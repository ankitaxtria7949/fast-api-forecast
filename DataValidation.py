import pandas as pd
def validate_data(df):
    error_list = []

    # Iterate over the months and validate
    months = df.columns[3:]  # Months start from the 4th column onwards

    for index, row in df.iterrows():
        for month in months:
            value = row[month]

            # Initialize the error reason
            reason = None

            # Check for invalid values (non-numeric or error values)
            if isinstance(value, str) and value.strip().startswith("#"):
                reason = "#VALUE! error"
            elif not isinstance(value, (int, float)):
                reason = "Invalid value (non-numeric)"
            elif value < 0:
                reason = "Negative value"
            elif any(char in str(value) for char in ["-", "#"]):
                reason = "Special character in value"

            # If there is an error, add it to the error list
            if reason:
                error_list.append({
                    "Country": row["Country"],
                    "Product": row["Product"],
                    "Forecast Scenario": row["Forecast Scenario"],
                    "Month": month,
                    "Value": value,
                    "Reason": reason
                })

    # Return the DataFrame of errors
    return pd.DataFrame(error_list)

