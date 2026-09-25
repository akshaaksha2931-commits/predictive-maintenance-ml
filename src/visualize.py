import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Load data
train = pd.read_csv("data/train.csv")

# -----------------------------
# 1. Machine Failure Distribution
# -----------------------------
plt.figure(figsize=(7, 5))

sns.countplot(
    data=train,
    x="Machine failure"
)

plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure")
plt.ylabel("Number of Machines")
plt.tight_layout()

plt.savefig("outputs/failure_distribution.png", dpi=300)
plt.close()


# -----------------------------
# 2. Failure by Machine Type
# -----------------------------
failure_by_type = (
    train.groupby("Type")["Machine failure"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(7, 5))

sns.barplot(
    data=failure_by_type,
    x="Type",
    y="Machine failure"
)

plt.title("Failure Rate by Machine Type")
plt.xlabel("Machine Type")
plt.ylabel("Failure Rate")

plt.tight_layout()

plt.savefig("outputs/failure_by_type.png", dpi=300)
plt.close()


# -----------------------------
# 3. Tool Wear vs Machine Failure
# -----------------------------
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=train,
    x="Machine failure",
    y="Tool wear [min]"
)

plt.title("Tool Wear vs Machine Failure")
plt.xlabel("Machine Failure")
plt.ylabel("Tool Wear [min]")

plt.tight_layout()

plt.savefig("outputs/tool_wear_vs_failure.png", dpi=300)
plt.close()


# -----------------------------
# 4. Correlation Heatmap
# -----------------------------
numeric_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure"
]

correlation = train[numeric_columns].corr()

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("outputs/correlation_heatmap.png", dpi=300)
plt.close()


print("All visualizations saved successfully!")

print("\nGenerated files:")
print("1. outputs/failure_distribution.png")
print("2. outputs/failure_by_type.png")
print("3. outputs/tool_wear_vs_failure.png")
print("4. outputs/correlation_heatmap.png")