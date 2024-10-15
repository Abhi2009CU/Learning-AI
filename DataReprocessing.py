#Data Preprocessing
import pandas as pd
import numpy as np

# Load a CSV file
data = pd.read_csv('sample data.csv')

# Handle missing values
data.fillna(0, inplace=True)
print(data)
# Normalize data
data = (data - data.mean()) / data.std()
print('\n\n\n')
print(data)
