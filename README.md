# weather-data-Analizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from datetime import datetime
📁 1. Data Collection & Preprocessing
python
Copy code
# Load the data
df = pd.read_csv("weather_data.csv")

# Check initial data
print(df.head())

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Normalize data (if needed for ML)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['temperature', 'humidity', 'rainfall']] = scaler.fit_transform(df[['temperature', 'humidity', 'rainfall']])
📊 2. Exploratory Data Analysis (EDA)
python
Copy code
# Descriptive Statistics
print(df[['temperature', 'humidity', 'rainfall']].describe())

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[['temperature', 'humidity', 'rainfall']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation between Weather Attributes")
plt.show()
📈 3. Visualizations
Line Chart: Temperature Trends Over Time
python
Copy code
plt.figure(figsize=(10,6))
sns.lineplot(data=df, x='year', y='temperature', ci=None)
plt.title("Yearly Average Temperature Trend")
plt.xlabel("Year")
plt.ylabel("Normalized Temperature")
plt.show()
Bar Graph: Rainfall Comparison by Year
python
Copy code
rain_by_year = df.groupby('year')['rainfall'].sum().reset_index()

plt.figure(figsize=(10,6))
sns.barplot(x='year', y='rainfall', data=rain_by_year, palette='Blues_d')
plt.title("Yearly Total Rainfall")
plt.xlabel("Year")
plt.ylabel("Total Rainfall (Normalized)")
plt.show()
Scatter Plot: Temperature vs. Humidity
python
Copy code
plt.figure(figsize=(8,6))
sns.scatterplot(x='temperature', y='humidity', data=df)
plt.title("Temperature vs Humidity")
plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.show()
🤖 4. Predictive Modelling with Linear Regression
Prepare Dataset for Regression
python
Copy code
# We'll predict average temperature by year
temp_by_year = df.groupby('year')['temperature'].mean().reset_index()

X = temp_by_year[['year']]
y = temp_by_year['temperature']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Model Evaluation:\nMSE: {mse:.4f}, RMSE: {rmse:.4f}")
Visualize Predictions
python
Copy code
# Predict for all available years and some future years
future_years = pd.DataFrame({'year': np.arange(df['year'].min(), df['year'].max()+10)})
future_preds = model.predict(future_years)

plt.figure(figsize=(10,6))
sns.scatterplot(x='year', y='temperature', data=temp_by_year, label="Actual")
plt.plot(future_years, future_preds, color='orange', label="Predicted Trend")
plt.title("Temperature Forecast Using Linear Regression")
plt.xlabel("Year")
plt.ylabel("Normalized Temperature")
plt.legend()
plt.show()
📋 5. Expected Output Summary
After running the full script:

✅ Console Output:

Descriptive statistics

Correlation matrix

Model accuracy (MSE, RMSE)

✅ Plots:

📈 Line chart: Yearly temperature trends

📊 Bar graph: Annual rainfall totals

🔵 Scatter plot: Temperature vs. Humidity

🟠 Trend line: Linear Regression temperature forecas
