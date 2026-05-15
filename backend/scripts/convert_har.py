import pandas as pd

# Load raw HAR data
X = pd.read_csv("../data/UCI HAR Dataset/train/X_train.txt", sep="\s+", header=None)

# Pick ONLY 9 features (simulate your sensor format)
selected = X.iloc[:, :9]

selected.columns = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z"
]

# Save CSV
selected.to_csv("har_sample.csv", index=False)

print("✅ Saved har_sample.csv")