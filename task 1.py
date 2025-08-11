# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Sample dataset (you can replace this with real data)
data = {
    'Square_Footage': [1500, 2000, 2500, 1800, 2200, 3000, 3500, 2700, 3200, 4000],
    'Bedrooms':       [3, 4, 4, 3, 4, 5, 5, 4, 5, 6],
    'Bathrooms':      [2, 3, 3, 2, 3, 4, 4, 3, 4, 5],
    'Price':          [300000, 400000, 450000, 350000, 420000, 500000, 600000, 480000, 550000, 650000]
}
# Create DataFrame
df = pd.DataFrame(data)

# Features (X) and target (y)
X = df[['Square_Footage', 'Bedrooms', 'Bathrooms']]
y = df['Price']
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Show model coefficients
print("\nIntercept:", model.intercept_)
print("Coefficients:", list(zip(X.columns, model.coef_)))

# Example prediction
example = pd.DataFrame({'Square_Footage': [2800], 'Bedrooms': [4], 'Bathrooms': [3]})
predicted_price = model.predict(example)
print("\nPredicted Price for example:", predicted_price[0])