# train_model.py

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib

# 1. Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# 2. Split features & target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)


# 7. Save model & scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
