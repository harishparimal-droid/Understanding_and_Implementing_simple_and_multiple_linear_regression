import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Import and preprocess the dataset
df = pd.read_csv('Housing.csv')

# Convert categorical yes/no to 1/0
binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_columns:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# Furnishing status with three categories, convert to dummy variables
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

# Define features and target variable
X = df.drop('price', axis=1)
y = df['price']

# 2. Split data into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit a Linear Regression model using sklearn.linear_model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate model using MAE, MSE, R²
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.4f}")

# 5. Plot regression line (for one feature: 'area') and interpret coefficients
plt.scatter(X_test['area'], y_test, color='blue', label='Actual Price')
plt.scatter(X_test['area'], y_pred, color='red', label='Predicted Price', alpha=0.5)
plt.xlabel('Area (sq units)')
plt.ylabel('Price')
plt.title('Linear Regression: Actual vs Predicted Prices')
plt.legend()
plt.show()

# Interpret coefficients
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print("\nModel Coefficients:")
print(coefficients)

