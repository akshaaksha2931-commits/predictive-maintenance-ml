import pandas as pd
import joblib

# Load deployment-oriented model
model = joblib.load("models/deployment_model.pkl")

print("===== PREDICTIVE MAINTENANCE SYSTEM =====")
print("Sensor-based Machine Failure Prediction")

# Get machine inputs
machine_type = input("Machine Type (H/L/M): ").upper()

air_temp = float(input("Air Temperature [K]: "))
process_temp = float(input("Process Temperature [K]: "))
rotational_speed = float(input("Rotational Speed [rpm]: "))
torque = float(input("Torque [Nm]: "))
tool_wear = float(input("Tool Wear [min]: "))

# Create input dataframe
data = pd.DataFrame([{
    "Type": machine_type,
    "Air temperature [K]": air_temp,
    "Process temperature [K]": process_temp,
    "Rotational speed [rpm]": rotational_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear
}])

# Encode machine type
data = pd.get_dummies(
    data,
    columns=["Type"],
    drop_first=True
)

# Match training columns
expected_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_L",
    "Type_M"
]

data = data.reindex(
    columns=expected_columns,
    fill_value=0
)

# Make prediction using optimized threshold
probability = model.predict_proba(data)[0][1]

threshold = 0.45

if probability >= threshold:
    prediction = 1
else:
    prediction = 0

print("\n===== RESULT =====")

if prediction == 1:
    print("⚠️ MACHINE FAILURE RISK DETECTED")
else:
    print("✅ NO MACHINE FAILURE RISK DETECTED")

print(f"Predicted failure probability: {probability * 100:.2f}%")