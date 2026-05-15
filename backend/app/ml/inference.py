from app.ml.preprocess import preprocess_csv
import joblib
import pandas as pd

# Load model once
model = joblib.load("models/random_forest_model.pkl")

# ---------------- SINGLE PREDICTION ----------------
def predict_from_csv(file_path):
    X = preprocess_csv(file_path)
    pred = model.predict(X)

    return int(pred.argmax()), float(pred.max())


# ---------------- WINDOW GENERATOR ----------------
def generate_windows(data, window_size=50):
    for i in range(len(data) - window_size):
        yield data[i:i+window_size]


# ---------------- STREAMING ----------------
def stream_predictions(file_path, window_size=5):
    df = pd.read_csv(file_path)

    expected_columns = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]

    if not all(col in df.columns for col in expected_columns):
        raise ValueError("CSV missing required columns")

    data = df[expected_columns].values

    for window in generate_windows(data, window_size):
        X = window.reshape(1, window.shape[0], window.shape[1])

        pred = model.predict(X)

        yield int(pred.argmax()), float(pred.max())