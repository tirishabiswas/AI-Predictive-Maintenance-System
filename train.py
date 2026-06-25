import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/predictive.csv")

# Convert Type column
df["Type"] = df["Type"].map({
    "L": 0,
    "M": 1,
    "H": 2
})

# Remove unnecessary columns and leakage columns
df = df.drop([
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
], axis=1)

# Features
X = df.drop("Machine failure", axis=1)

# Target
y = df["Machine failure"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "model/predictive_model.pkl")

print("Model Saved Successfully!")