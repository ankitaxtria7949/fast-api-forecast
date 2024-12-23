import pandas as pd
def validate_data(df):
    months = df.columns[4:]  # Months start from the 4th column onwards
    error_list = []
    for col in months:
        if df[col].dtype == 'object':  # Check if the column is of object type
            try:
                # Remove commas and convert to float
                df[col] = df[col].str.replace(',', '').astype(float)
            except ValueError:
                pass

    print(df.info())
    for index, row in df.iterrows():
        for month in months:
            value = row[month]
            reason = None
            # Check for invalid values (non-numeric or error values)
            if isinstance(value, str) and value.strip().startswith("#"):
                reason = "#VALUE! error"
            elif any(char in str(value) for char in ["-"]):
                reason = "Negative value"
            elif any(char in str(value) for char in ["#", "@"]):
                reason = "Special character in value"

            # If there is an error, add it to the error list
            if reason:
                error_list.append({
                    "Product": row["Product"],
                    "Country": row["Country"],
                    "Forecast Scenario": row["Forecast Scenario"],
                    "Months": month,
                    "Value": value,
                    "Reason": reason
                })

    # Return the DataFrame of errors
    return pd.DataFrame(error_list)