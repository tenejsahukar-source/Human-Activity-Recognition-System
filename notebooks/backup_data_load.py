import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd

features = pd.read_csv("../data/UCI HAR Dataset/features.txt", sep=r"\s+", header=None)[1].values

X_train = pd.read_csv("../data/UCI HAR Dataset/train/X_train.txt", sep=r"\s+", header=None)
X_train.columns = features

y_train = pd.read_csv("../data/UCI HAR Dataset/train/y_train.txt", header=None)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print(X_train.head())

# Load activity labels
activity_labels = pd.read_csv(
    "../data/UCI HAR Dataset/activity_labels.txt",
    sep=r"\s+",
    header=None,
    names=["id", "activity"]
)

# Convert to dictionary
activity_map = dict(zip(activity_labels.id, activity_labels.activity))

# Map numbers to labels
y_train["activity"] = y_train[0].map(activity_map)

print("\nActivity Mapping Sample:")
print(y_train.head())

# Count of each activity
counts = y_train["activity"].value_counts()

print("\nActivity Distribution:")
print(counts)

# Plot
plt.figure(figsize=(8,5))
counts.plot(kind="bar")
plt.title("Activity Distribution")
plt.xlabel("Activity")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
#plt.show()

# Drop numeric label column
y = y_train["activity"]

# Split data
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_split, y_train_split)

y_pred = model.predict(X_val)

print("\nAccuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_val, y_pred))

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_split, y_train_split)

rf_pred = rf_model.predict(X_val)

print("\n===== RANDOM FOREST =====")
print("Accuracy:", accuracy_score(y_val, rf_pred))
print("\nClassification Report:\n")
print(classification_report(y_val, rf_pred))

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# ================= FEATURE IMPORTANCE =================

import numpy as np

importances = rf_model.feature_importances_
indices = np.argsort(importances)[-10:]

plt.figure(figsize=(8,5))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.title("Top 10 Important Features")
plt.show()

# ================= LSTM MODEL (IMPROVED) =================

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# ================= CORRECT LSTM (REAL ACCURACY ~80%) =================

from tensorflow.keras.callbacks import EarlyStopping

# Load signals (train/test separately)
def load_signals(split):
    signals = []
    signal_types = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]

    for signal in signal_types:
        data = pd.read_csv(
            f"../data/UCI HAR Dataset/{split}/Inertial Signals/{signal}_{split}.txt",
            sep=r"\s+",
            header=None
        )
        signals.append(data.values)

    return np.transpose(np.array(signals), (1, 2, 0))


# 🔥 LOAD TRAIN & TEST (NO LEAKAGE)
X_train_lstm = load_signals("train")
X_test_lstm = load_signals("test")

y_train_lstm = pd.read_csv(
    "../data/UCI HAR Dataset/train/y_train.txt", header=None
)[0] - 1

y_test_lstm = pd.read_csv(
    "../data/UCI HAR Dataset/test/y_test.txt", header=None
)[0] - 1

y_train_lstm = to_categorical(y_train_lstm)
y_test_lstm = to_categorical(y_test_lstm)

print("Train Shape:", X_train_lstm.shape)
print("Test Shape:", X_test_lstm.shape)


# 🔥 NORMALIZATION (fit on train only)
mean = np.mean(X_train_lstm, axis=0)
std = np.std(X_train_lstm, axis=0) + 1e-8

X_train_lstm = (X_train_lstm - mean) / std
X_test_lstm = (X_test_lstm - mean) / std


# 🔥 MODEL
model_lstm = Sequential()

model_lstm.add(LSTM(128, return_sequences=True, input_shape=(128, 9)))
model_lstm.add(Dropout(0.2))

model_lstm.add(LSTM(64))
model_lstm.add(Dense(64, activation='relu'))
model_lstm.add(Dense(6, activation='softmax'))

model_lstm.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# 🔥 TRAIN
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

print("\n===== TRAINING LSTM =====")
model_lstm.fit(
    X_train_lstm,
    y_train_lstm,
    epochs=25,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop]
)


# 🔥 REAL TEST RESULT
loss, acc = model_lstm.evaluate(X_test_lstm, y_test_lstm)

print("\n===== LSTM TEST RESULT =====")
print("Accuracy:", acc)


# ================= CNN + LSTM MODEL =================

from tensorflow.keras.layers import Conv1D, MaxPooling1D

model_cnn_lstm = Sequential()

# CNN part (feature extraction)
model_cnn_lstm.add(Conv1D(
    filters=64,
    kernel_size=3,
    activation='relu',
    input_shape=(128, 9)
))
model_cnn_lstm.add(MaxPooling1D(pool_size=2))

# LSTM part (sequence learning)
model_cnn_lstm.add(LSTM(64))

# Dense layers
model_cnn_lstm.add(Dense(64, activation='relu'))
model_cnn_lstm.add(Dense(6, activation='softmax'))

model_cnn_lstm.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\n===== TRAINING CNN + LSTM =====")

early_stop_cnn = EarlyStopping(patience=5, restore_best_weights=True)

history_cnn = model_cnn_lstm.fit(
    X_train_lstm,
    y_train_lstm,
    epochs=25,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop_cnn]
)

# Evaluate on TEST (IMPORTANT)
loss_cnn, acc_cnn = model_cnn_lstm.evaluate(X_test_lstm, y_test_lstm)

print("\n===== CNN + LSTM TEST RESULT =====")
print("Accuracy:", acc_cnn)

