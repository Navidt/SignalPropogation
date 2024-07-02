from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load the merged data with 271 columns
merged_data = pd.read_csv('merged_data.csv')

# Split the data into features (X) and target variable (y)
X = merged_data.iloc[:, :-1]  # Features (all columns except the last one)
y = merged_data.iloc[:, -1]   # Target variable (moisture value, last column)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

