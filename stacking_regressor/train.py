import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load Dataset
df = pd.read_csv("data/insurance.csv")

# Encode Categorical Columns
encoders = {}

for col in ["sex", "smoker", "region"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features and Target
X = df.drop("charges", axis=1)
y = df["charges"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Base Models
estimators = [
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ("gb", GradientBoostingRegressor(random_state=42))
]

# Stacking Regressor
stack_model = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression()
)

# Train
stack_model.fit(X_train, y_train)

# Prediction
pred = stack_model.predict(X_test)

print("R2 Score:", r2_score(y_test, pred))

# Create models folder
os.makedirs("models", exist_ok=True)

# Save Model
pickle.dump(
    stack_model,
    open("models/stacking_model.pkl", "wb")
)

# Save Encoders
pickle.dump(
    encoders,
    open("models/encoders.pkl", "wb")
)

print("Model Saved Successfully")