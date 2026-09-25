import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Load data
df = pd.read_csv("data/train.csv")

# Deployment features only
features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Machine failure"]

# Encode Type
X = pd.get_dummies(X, columns=["Type"], drop_first=True)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load trained deployment model
model = joblib.load("models/deployment_model.pkl")

# Get failure probabilities
probabilities = model.predict_proba(X_val)[:, 1]

# Test different thresholds
thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

print("\n===== THRESHOLD ANALYSIS =====")

for threshold in thresholds:

    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(y_val, predictions, zero_division=0)
    recall = recall_score(y_val, predictions, zero_division=0)
    f1 = f1_score(y_val, predictions, zero_division=0)

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )