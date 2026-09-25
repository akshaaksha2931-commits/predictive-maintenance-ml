import pandas as pd

# Load the dataset
train_path = "data/train.csv"
test_path = "data/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Display basic information
print("Train shape:", train.shape)
print("Test shape:", test.shape)

print("\nTrain columns:")
print(train.columns.tolist())

print("\nFirst 5 rows:")
print(train.head())