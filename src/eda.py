import pandas as pd

# Load training data
train = pd.read_csv("data/train.csv")

print("===== DATASET OVERVIEW =====")
print("Shape:", train.shape)

print("\n===== MISSING VALUES =====")
print(train.isnull().sum())

print("\n===== DUPLICATES =====")
print("Duplicate rows:", train.duplicated().sum())

print("\n===== MACHINE FAILURE =====")
print(train["Machine failure"].value_counts())

print("\nFailure percentage:")
print(train["Machine failure"].value_counts(normalize=True) * 100)

print("\n===== MACHINE TYPE =====")
print(train["Type"].value_counts())

print("\n===== FAILURE BY MACHINE TYPE =====")
print(
    train.groupby("Type")["Machine failure"]
    .agg(["count", "sum", "mean"])
)

# Failure indicator analysis
failure_indicators = ["TWF", "HDF", "PWF", "OSF", "RNF"]

print("\n===== FAILURE INDICATORS =====")

for column in failure_indicators:
    print(f"\n{column}:")
    print(train[column].value_counts())

print("\n===== FAILURE INDICATOR COUNTS =====")

for column in failure_indicators:
    print(f"{column}: {train[column].sum()}")

# Tool wear analysis
print("\n===== TOOL WEAR =====")
print(train["Tool wear [min]"].describe())

# Create tool-wear bands
train["wear_band"] = pd.cut(
    train["Tool wear [min]"],
    bins=[0, 50, 100, 150, 200, float("inf")],
    labels=["0-50", "51-100", "101-150", "151-200", "200+"]
)

print("\n===== FAILURE RATE BY TOOL WEAR BAND =====")

print(
    train.groupby(
        "wear_band",
        observed=True
    )["Machine failure"]
    .agg(["count", "sum", "mean"])
)