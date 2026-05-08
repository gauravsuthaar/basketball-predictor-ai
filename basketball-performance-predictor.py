# Import pandas for handling datasets/tables
import pandas as pd

# Import function to split data into training and testing
from sklearn.model_selection import train_test_split

# Import Linear Regression model
from sklearn.linear_model import LinearRegression


# Load CSV dataset into a dataframe
df = pd.read_csv("player_stats.csv")


# Print dataset to verify it loaded correctly
print(df)


# FEATURES (X)
# These are the input columns used for prediction
X = df[[
    "practice_hours",
    "sleep_hours",
    "shots_attempted",
    "stamina_score"
]]


# TARGET (y)
# This is what we want the model to predict
y = df["points_scored"]


# Split data into training and testing sets
# 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create Linear Regression model
model = LinearRegression()


# Train the model using training data
model.fit(X_train, y_train)


# Give new player stats for prediction
# Format:
# [practice_hours, sleep_hours, shots_attempted, stamina_score]
prediction = model.predict([[5, 8, 65, 78]])


# Print predicted points
# prediction[0] means first predicted value from returned array
print("Predicted Points:", prediction[0])