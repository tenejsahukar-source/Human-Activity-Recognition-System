import pandas as pd
import numpy as np

def preprocess_csv(file_path):
    df = pd.read_csv(file_path)

    # Expected features (must match training)
    expected_columns = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]

    # Check columns
    if not all(col in df.columns for col in expected_columns):
        raise ValueError("CSV missing required columns")

    data = df[expected_columns].values

    # reshape for LSTM → (1, timesteps, features)
    X = data.reshape(1, data.shape[0], data.shape[1])

    return X