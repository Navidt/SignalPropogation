import pandas as pd

# Load the CSV files
csi_data = pd.read_csv('./data/csi_data2.csv')
moisture = pd.read_csv('./data/moisture2.csv')

# Interpolate the moisture data to match the length of csi_data
moisture_interpolated = moisture.interpolate(method='linear')

# Ensure that the moisture data has the same number of rows as csi_data
moisture_interpolated = moisture_interpolated.reindex(csi_data.index, method='ffill')

# Merge the two dataframes
merged_data = pd.concat([csi_data, moisture_interpolated], axis=1)

# Write the merged dataframe to a new CSV file
merged_data.to_csv('merged_data2.csv', index=False)
