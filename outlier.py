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
    final_df['IQRUCL'] = final_df['6-Months Rolling Q3'] + 1.5 * final_df['IQR']
 
    # Calculate the LCL (Lower Control Limit) = Q1 - 1.5 * IQR
    final_df['IQRLCL'] = final_df['6-Months Rolling Q1'] - 1.5 * final_df['IQR']
 
    # Set LCL to 0 if it's negative
    final_df['IQRLCL'] = final_df['IQRLCL'].apply(lambda x: max(x, 0))
 
    # Add IQR Outlier Flag
    def get_outlier_flag(row):
        if row['Market Volume'] < row['IQRLCL'] or row['Market Volume'] > row['IQRUCL']:
            return 'Outlier'
        else:
            return 'Not Outlier'
 
    final_df['IQR Outlier Flag'] = final_df.apply(get_outlier_flag, axis=1)
 
    # Calculate Z-Score
    def calculate_z_score(row):
        mean = row['6-Months Rolling Avg']
        std = row['6-Months Rolling Std']
        return (row['Market Volume'] - mean) / std if std != 0 else 0
   
    final_df['Z-Score'] = final_df.apply(calculate_z_score, axis=1)
 
    # Calculate Z-UCL and Z-LCL
    final_df['ZUCL'] = final_df['6-Months Rolling Avg'] + (3 * final_df['6-Months Rolling Std'])
    final_df['ZLCL'] = final_df['6-Months Rolling Avg'] - (3 * final_df['6-Months Rolling Std'])
    final_df['ZLCL'] = final_df['ZLCL'].apply(lambda x: max(x, 0))  # Set Z-LCL to 0 if negative
 
    # Add Z-Score Outlier Flag
    def get_zscore_outlier_flag(row):
        if row['Market Volume'] < row['ZLCL'] or row['Market Volume'] > row['ZUCL']:
            return 'Outlier'
        else:
            return 'Not Outlier'
 
    final_df['Z-Score Outlier Flag'] = final_df.apply(get_zscore_outlier_flag, axis=1)
 
    final_df['6-Months Rolling Median'] = final_df.groupby('Primary key')['Previous Market Volume'].rolling(6, min_periods=1).median().reset_index(level=0, drop=True)
 
    final_df['Absolute Deviation'] = abs(final_df['Market Volume'] - final_df['6-Months Rolling Median'])
 
    #Calculate the 6-Months Rolling MAD
    final_df['6-Months Rolling MAD'] = final_df.groupby('Primary key')['Absolute Deviation'].transform(
        lambda x: x.rolling(window=6, min_periods=1).median()
    )
 
    hampel_threshold = 3  # Typical value for Hampel method
    final_df['HUCL'] = final_df['6-Months Rolling Median'] + hampel_threshold * final_df['6-Months Rolling MAD']
    final_df['HLCL'] = final_df['6-Months Rolling Median'] - hampel_threshold * final_df['6-Months Rolling MAD']
    # Set Hampel LCL to 0 if it's negative
    final_df['HLCL'] = final_df['HLCL'].apply(lambda x: max(x, 0))
 
    def get_hample_outlier_flag(row):
            if row['Market Volume'] < row['HLCL'] or row['Market Volume'] > row['HUCL']:
                return 'Outlier'
            else:
                return 'Not Outlier'
 
    final_df['Hample Outlier Flag'] = final_df.apply(get_hample_outlier_flag, axis=1)
 
 
    # Drop the 'Previous Market Volume' column, as it's no longer needed
    final_df.drop(columns=['Previous Market Volume'], inplace=True)
 
    # Separate Outliers for IQR and Z-Score
    iqr_outliers = final_df[final_df['IQR Outlier Flag'] == 'Outlier']
    zscore_outliers = final_df[final_df['Z-Score Outlier Flag'] == 'Outlier']
    hample_outliers = final_df[final_df['Hample Outlier Flag'] == 'Outlier']

    # Filter out rows where Market Volume is within ±5% of LCL and UCL
    iqr_outliers = iqr_outliers[~((iqr_outliers['Market Volume'] >= 0.95 * iqr_outliers['IQRLCL']) & (iqr_outliers['Market Volume'] <= 1.05 * iqr_outliers['IQRUCL']))]
    zscore_outliers = zscore_outliers[~((zscore_outliers['Market Volume'] >= 0.95 * zscore_outliers['ZLCL']) & (zscore_outliers['Market Volume'] <= 1.05 * zscore_outliers['ZUCL']))]
    hample_outliers = hample_outliers[~((hample_outliers['Market Volume'] >= 0.95 * hample_outliers['HLCL']) & (hample_outliers['Market Volume'] <= 1.05 * hample_outliers['HUCL']))]

    common_outliers = pd.merge(iqr_outliers, zscore_outliers, how='inner')
    common_outliers = pd.merge(common_outliers, hample_outliers, how='inner')

    # Combine all unique outliers
    union_outliers = pd.concat([iqr_outliers, zscore_outliers, hample_outliers, common_outliers]).drop_duplicates()
    union_outliers = union_outliers[["Product", "Country", "Forecast Scenario", "Months", "Market Volume", "ZLCL", "ZUCL"]]
 
    return union_outliers, final_df


