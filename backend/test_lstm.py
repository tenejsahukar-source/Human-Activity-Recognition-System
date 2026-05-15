from lstm_utils import load_single_sample
from tensorflow.keras.models import load_model
import numpy as np

# Load sample
sample = load_single_sample(
    "../data/UCI HAR Dataset/test/Inertial Signals",
    index=0
)

# Load model
model = load_model("models/lstm_har_model.h5")
# Reshape for LSTM
sample = sample.reshape(1, 128, 9)

# Predict
pred = model.predict(sample)
pred_class = np.argmax(pred)

LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

print("Prediction:", LABELS[pred_class])
print("Confidence:", np.max(pred))