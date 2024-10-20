from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Load a CSV file
data = pd.read_csv('sample_data.csv')

# Handle missing values
data.fillna(0, inplace=True)

# Normalize data
data = (data - data.mean()) / data.std()


# Load data
X = data[['SessionsDate']]  # Features
y = data['Website Views']  # Target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(predictions, mse, r2)

