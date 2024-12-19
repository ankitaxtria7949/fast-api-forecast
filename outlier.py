import pandas as pd


def detect_outliers_by_month(df):
    # Initialize an empty list to collect the results
    melted_list = []

    # Loop through each row of the DataFrame
    for i, row in df.iterrows():
        # Melt the row (convert from wide to long format)
        df_melted = pd.melt(row.to_frame().T, 
                            id_vars=['Product', 'Country', 'Forecast Scenario', 'Product life cycle stage'], 
                            var_name='Date', 
                            value_name='Market Volume')

        # Create the 'Primary key' column by concatenating Product, Country, and Forecast Scenario
        df_melted['Primary key'] = df_melted['Product'] + df_melted['Country'] + df_melted['Forecast Scenario']

        # Convert 'Date' to datetime
        df_melted['Date'] = pd.to_datetime(df_melted['Date'], format='%Y-%m-%d %H:%M:%S')

        # Extract the Month-Year from the 'Date' column
        df_melted['Months'] = df_melted['Date'].dt.strftime('%b-%y')

        # Drop rows where 'Market Volume' is NaN
        df_melted = df_melted.dropna(subset=['Market Volume'])

        # Append the melted DataFrame to the list
        melted_list.append(df_melted)

    # Concatenate all the melted DataFrames into one DataFrame
    final_df = pd.concat(melted_list, ignore_index=True)

    # Display the reshaped DataFrame with the proper columns
    final_df = final_df[['Product', 'Country', 'Forecast Scenario', 'Product life cycle stage', 'Primary key', 'Months', 'Market Volume']]

    # Shift the Market Volume by one to exclude the current value for the rolling calculations
    final_df['Previous Market Volume'] = final_df.groupby('Primary key')['Market Volume'].shift(1)

    # Calculate the rolling average for the previous 6 months (excluding the current month)
    final_df['6-Months Rolling Avg'] = final_df.groupby('Primary key')['Previous Market Volume'].rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)

    # Calculate the rolling standard deviation for the previous 6 months (excluding the current month)
    final_df['6-Months Rolling Std'] = final_df.groupby('Primary key')['Previous Market Volume'].rolling(6, min_periods=1).std().reset_index(level=0, drop=True)

    # Calculate the rolling Q1 (25th percentile) for the previous 6 months (excluding the current month)
    final_df['6-Months Rolling Q1'] = final_df.groupby('Primary key')['Previous Market Volume'].rolling(6, min_periods=1).quantile(0.25).reset_index(level=0, drop=True)

    # Calculate the rolling Q3 (75th percentile) for the previous 6 months (excluding the current month)
    final_df['6-Months Rolling Q3'] = final_df.groupby('Primary key')['Previous Market Volume'].rolling(6, min_periods=1).quantile(0.75).reset_index(level=0, drop=True)

    # Calculate the IQR (Interquartile Range)
    final_df['IQR'] = final_df['6-Months Rolling Q3'] - final_df['6-Months Rolling Q1']

    # Calculate the UCL (Upper Control Limit) = Q3 + 1.5 * IQR
    final_df['UCL'] = final_df['6-Months Rolling Q3'] + 1.5 * final_df['IQR']

    # Calculate the LCL (Lower Control Limit) = Q1 - 1.5 * IQR
    final_df['LCL'] = final_df['6-Months Rolling Q1'] - 1.5 * final_df['IQR']

    # Set LCL to 0 if it's negative
    final_df['LCL'] = final_df['LCL'].apply(lambda x: max(x, 0))

    # Define the function to calculate the Z-score
    def calculate_z_score(row):
        mean = row['6-Months Rolling Avg']
        std = row['6-Months Rolling Std']
        return (row['Market Volume'] - mean) / std if std != 0 else 0

    # Add an Outlier Flag based on whether Market Volume is outside of LCL and UCL or based on Z-score
    def get_outlier_flag(row):
        if row['Product life cycle stage'] == 'Mature':
            # Z-score method for 'Mature' product life cycle stage
            z_score = calculate_z_score(row)
            if abs(z_score) > 3:
                return 'Outlier'
            else:
                return 'Not Outlier'
        else:
            # IQR method for other product life cycle stages
            if row['Market Volume'] < row['LCL'] or row['Market Volume'] > row['UCL']:
                return 'Outlier'
            else:
                return 'Not Outlier'

    # Apply the Outlier Flag function to each row
    final_df['Outlier Flag'] = final_df.apply(get_outlier_flag, axis=1)

    # Drop the 'Previous Market Volume' column, as it's no longer needed
    final_df.drop(columns=['Previous Market Volume'], inplace=True)
    final_df2 = final_df[final_df['Outlier Flag'] == 'Outlier']

    final_df2['Deviation(%)'] = final_df2.apply(
    lambda row: row['Market Volume'] - row['UCL'] 
    if row['Market Volume'] > row['UCL'] 
    else row['LCL'] - row['Market Volume'] 
    if row['Market Volume'] < row['LCL'] 
    else 0,
    axis=1
)
    # Show the result
    final_df2 = final_df2[['Product', 'Country', 'Forecast Scenario', 'Product life cycle stage',  'Months', 'Market Volume', 'UCL', 'LCL', 'Deviation(%)']]


    return final_df2, final_df


