import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/machine_failure_model.pkl")

print("===== MACHINE FAILURE PREDICTION =====")

# Get machine information
machine_type = input("Machine Type (H/L/M): ").upper()

air_temp = float(input("Air Temperature [K]: "))
process_temp = float(input("Process Temperature [K]: "))
rotational_speed = float(input("Rotational Speed [rpm]: "))
torque = float(input("Torque [Nm]: "))
tool_wear = float(input("Tool Wear [min]: "))

# Failure indicators
twf = int(input("TWF (0/1): "))
hdf = int(input("HDF (0/1): "))
pwf = int(input("PWF (0/1): "))
osf = int(input("OSF (0/1): "))
rnf = int(input("RNF (0/1): "))

# Create input dataframe
data = pd.DataFrame([{
    "Type": machine_type,
    "Air temperature [K]": air_temp,
    "Process temperature [K]": process_temp,
    "Rotational speed [rpm]": rotational_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear,
    "TWF": twf,
    "HDF": hdf,
    "PWF": pwf,
    "OSF": osf,
    "RNF": rnf
}])

# Encode Type exactly like training
data = pd.get_dummies(
    data,
    columns=["Type"],
    drop_first=True
)

# Make sure columns match training columns
expected_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
    "Type_L",
    "Type_M"
]

data = data.reindex(columns=expected_columns, fill_value=0)

# Prediction
prediction = model.predict(data)[0]
probability = model.predict_proba(data)[0][1]

print("\n===== RESULT =====")

if prediction == 1:
    print("⚠️ MACHINE FAILURE PREDICTED")
else:
    print("✅ NO MACHINE FAILURE PREDICTED")

print(f"Failure probability: {probability * 100:.2f}%")