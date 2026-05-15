"""
Human Activity Recognition (HAR) - Complete Pipeline
Uses UCI HAR Dataset with ML (Logistic Regression, Random Forest) and DL (LSTM, CNN+LSTM)
✅ Proper data leakage prevention
✅ Official train/test split
✅ Comprehensive visualization and comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, precision_recall_fscore_support)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ================= 1. LOAD ENGINEERED FEATURES =================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

features = pd.read_csv("../data/UCI HAR Dataset/features.txt", sep=r"\s+", header=None)[1].values

# Load TRAIN engineered features
X_train = pd.read_csv("../data/UCI HAR Dataset/train/X_train.txt", sep=r"\s+", header=None)
X_train.columns = features

# Load TEST engineered features (OFFICIAL)
X_test = pd.read_csv("../data/UCI HAR Dataset/test/X_test.txt", sep=r"\s+", header=None)
X_test.columns = features

# Load activity labels mapping
activity_labels = pd.read_csv(
    "../data/UCI HAR Dataset/activity_labels.txt",
    sep=r"\s+", header=None, names=["id", "activity"]
)
activity_map = dict(zip(activity_labels.id, activity_labels.activity))
activity_labels_list = sorted(activity_map.values())

# Load TRAIN labels
y_train = pd.read_csv("../data/UCI HAR Dataset/train/y_train.txt", header=None)
y_train["activity"] = y_train[0].map(activity_map)
y_train = y_train["activity"]

# Load TEST labels (OFFICIAL)
y_test = pd.read_csv("../data/UCI HAR Dataset/test/y_test.txt", header=None)
y_test["activity"] = y_test[0].map(activity_map)
y_test = y_test["activity"]

print(f"\n✅ Train shape: {X_train.shape}")
print(f"✅ Test shape: {X_test.shape}")
print(f"\n📊 Activity distribution:")
print(y_train.value_counts().sort_index())

# ================= 2. MACHINE LEARNING MODELS =================

print("\n" + "=" * 60)
print("MACHINE LEARNING MODELS")
print("=" * 60)

# Logistic Regression
print("\n>>> LOGISTIC REGRESSION")
lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)

print(f"✅ Accuracy: {acc_lr:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

# Random Forest
print("\n>>> RANDOM FOREST")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"✅ Accuracy: {acc_rf:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Feature Importance
print("\n>>> TOP 10 IMPORTANT FEATURES")
importances = rf_model.feature_importances_
top_indices = np.argsort(importances)[-10:]
for rank, idx in enumerate(reversed(top_indices), 1):
    print(f"{rank}. {features[idx]}: {importances[idx]:.6f}")

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_indices)), importances[top_indices], color='steelblue')
plt.yticks(range(len(top_indices)), [features[i] for i in top_indices])
plt.xlabel("Importance Score", fontsize=12)
plt.title("Top 10 Important Features (Random Forest)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# ================= 3. LOAD RAW TIME-SERIES SIGNALS FOR DEEP LEARNING =================

def load_signals(split):
    """Load raw sensor signals (128 timesteps × 9 signals)"""
    signals = []
    signal_types = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]
    
    for signal in signal_types:
        data = pd.read_csv(
            f"../data/UCI HAR Dataset/{split}/Inertial Signals/{signal}_{split}.txt",
            sep=r"\s+", header=None
        )
        signals.append(data.values)
    
    # Shape: (n_samples, n_timesteps, n_signals)
    return np.transpose(np.array(signals), (1, 2, 0))


print("\n" + "=" * 60)
print("DEEP LEARNING - LOADING SIGNALS")
print("=" * 60)

X_train_lstm = load_signals("train")
X_test_lstm = load_signals("test")

y_train_lstm = pd.read_csv("../data/UCI HAR Dataset/train/y_train.txt", header=None)[0] - 1
y_test_lstm = pd.read_csv("../data/UCI HAR Dataset/test/y_test.txt", header=None)[0] - 1

y_train_lstm_cat = to_categorical(y_train_lstm)
y_test_lstm_cat = to_categorical(y_test_lstm)

print(f"\n✅ Train signals shape: {X_train_lstm.shape}")
print(f"✅ Test signals shape: {X_test_lstm.shape}")
print(f"   (n_samples, timesteps, signals)")

# Normalization - FIT ON TRAIN ONLY (prevent data leakage)
mean = np.mean(X_train_lstm, axis=0)
std = np.std(X_train_lstm, axis=0) + 1e-8
X_train_lstm_norm = (X_train_lstm - mean) / std
X_test_lstm_norm = (X_test_lstm - mean) / std

print(f"✅ Data normalized (mean/std fit on training data only)")

# ================= 4. LSTM MODEL =================

print("\n" + "=" * 60)
print("DEEP LEARNING MODEL 1: LSTM")
print("=" * 60)

model_lstm = Sequential([
    LSTM(128, return_sequences=True, input_shape=(128, 9)),
    Dropout(0.2),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(6, activation='softmax')
])

model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("\n>>> TRAINING LSTM")
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

history_lstm = model_lstm.fit(
    X_train_lstm_norm, y_train_lstm_cat,
    epochs=25, batch_size=64, validation_split=0.2,
    callbacks=[early_stop], verbose=1
)

loss_lstm, acc_lstm = model_lstm.evaluate(X_test_lstm_norm, y_test_lstm_cat, verbose=0)
print(f"\n✅ LSTM Test Accuracy: {acc_lstm:.4f}")

# Get predictions
y_pred_lstm_proba = model_lstm.predict(X_test_lstm_norm, verbose=0)
y_pred_lstm = np.argmax(y_pred_lstm_proba, axis=1)

# ================= 5. CNN + LSTM MODEL =================

print("\n" + "=" * 60)
print("DEEP LEARNING MODEL 2: CNN + LSTM")
print("=" * 60)

model_cnn_lstm = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(128, 9)),
    MaxPooling1D(pool_size=2),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(6, activation='softmax')
])

model_cnn_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("\n>>> TRAINING CNN + LSTM")
early_stop_cnn = EarlyStopping(patience=5, restore_best_weights=True)

history_cnn_lstm = model_cnn_lstm.fit(
    X_train_lstm_norm, y_train_lstm_cat,
    epochs=25, batch_size=64, validation_split=0.2,
    callbacks=[early_stop_cnn], verbose=1
)

loss_cnn, acc_cnn = model_cnn_lstm.evaluate(X_test_lstm_norm, y_test_lstm_cat, verbose=0)
print(f"\n✅ CNN+LSTM Test Accuracy: {acc_cnn:.4f}")

# Get predictions
y_pred_cnn_proba = model_cnn_lstm.predict(X_test_lstm_norm, verbose=0)
y_pred_cnn = np.argmax(y_pred_cnn_proba, axis=1)

# ================= 6. CONFUSION MATRICES =================

print("\n" + "=" * 60)
print("CONFUSION MATRICES")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', ax=axes[0, 0], 
            xticklabels=activity_labels_list, yticklabels=activity_labels_list,
            cmap='Blues', cbar_kws={'label': 'Count'})
axes[0, 0].set_title(f"Logistic Regression (Acc: {acc_lr:.2%})", fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel("True Label", fontsize=11)
axes[0, 0].set_xlabel("Predicted Label", fontsize=11)

# Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', ax=axes[0, 1], 
            xticklabels=activity_labels_list, yticklabels=activity_labels_list,
            cmap='Blues', cbar_kws={'label': 'Count'})
axes[0, 1].set_title(f"Random Forest (Acc: {acc_rf:.2%})", fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel("True Label", fontsize=11)
axes[0, 1].set_xlabel("Predicted Label", fontsize=11)

# LSTM
cm_lstm = confusion_matrix(y_test_lstm, y_pred_lstm)
sns.heatmap(cm_lstm, annot=True, fmt='d', ax=axes[1, 0], 
            xticklabels=activity_labels_list, yticklabels=activity_labels_list,
            cmap='Blues', cbar_kws={'label': 'Count'})
axes[1, 0].set_title(f"LSTM (Acc: {acc_lstm:.2%})", fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel("True Label", fontsize=11)
axes[1, 0].set_xlabel("Predicted Label", fontsize=11)

# CNN+LSTM
cm_cnn = confusion_matrix(y_test_lstm, y_pred_cnn)
sns.heatmap(cm_cnn, annot=True, fmt='d', ax=axes[1, 1], 
            xticklabels=activity_labels_list, yticklabels=activity_labels_list,
            cmap='Blues', cbar_kws={'label': 'Count'})
axes[1, 1].set_title(f"CNN+LSTM (Acc: {acc_cnn:.2%})", fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel("True Label", fontsize=11)
axes[1, 1].set_xlabel("Predicted Label", fontsize=11)

plt.suptitle("Confusion Matrices - All Models", fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

# ================= 7. COMPREHENSIVE COMPARISON TABLE =================

print("\n" + "=" * 60)
print("MODEL COMPARISON - COMPREHENSIVE METRICS")
print("=" * 60)

def get_metrics(y_true, y_pred):
    """Calculate precision, recall, f1, accuracy"""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

comparison = pd.DataFrame({
    'Logistic Regression': get_metrics(y_test, y_pred_lr),
    'Random Forest': get_metrics(y_test, y_pred_rf),
    'LSTM': get_metrics(y_test_lstm, y_pred_lstm),
    'CNN+LSTM': get_metrics(y_test_lstm, y_pred_cnn)
}).T

print("\n", comparison.round(4))

# Save to CSV
comparison.to_csv('model_comparison.csv')
print("\n✅ Comparison saved to 'model_comparison.csv'")

# ================= 8. TRAINING HISTORY VISUALIZATION =================

print("\n" + "=" * 60)
print("TRAINING CURVES")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history_lstm.history['accuracy'], label='LSTM Train', linewidth=2)
axes[0].plot(history_lstm.history['val_accuracy'], label='LSTM Val', linewidth=2)
axes[0].plot(history_cnn_lstm.history['accuracy'], label='CNN+LSTM Train', linewidth=2)
axes[0].plot(history_cnn_lstm.history['val_accuracy'], label='CNN+LSTM Val', linewidth=2)
axes[0].set_title('Training Accuracy Over Epochs', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Accuracy', fontsize=11)
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Loss
axes[1].plot(history_lstm.history['loss'], label='LSTM Train', linewidth=2)
axes[1].plot(history_lstm.history['val_loss'], label='LSTM Val', linewidth=2)
axes[1].plot(history_cnn_lstm.history['loss'], label='CNN+LSTM Train', linewidth=2)
axes[1].plot(history_cnn_lstm.history['val_loss'], label='CNN+LSTM Val', linewidth=2)
axes[1].set_title('Training Loss Over Epochs', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Loss', fontsize=11)
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# ================= 9. PER-CLASS ANALYSIS =================

print("\n" + "=" * 60)
print("PER-CLASS PERFORMANCE ANALYSIS")
print("=" * 60)

def per_class_accuracy(y_true, y_pred, labels):
    results = {}
    for label in labels:
        mask = y_true == label
        acc = accuracy_score(y_true[mask], y_pred[mask]) if np.sum(mask) > 0 else 0
        results[label] = acc
    return results

print("\n>>> RANDOM FOREST - PER-CLASS ACCURACY")
# Convert labels → numbers (for RF)
import numpy as np

rf_per_class = per_class_accuracy(
    y_test.values,
    y_pred_rf,
    activity_labels_list
)

for activity, acc in sorted(rf_per_class.items(), key=lambda x: x[1]):
    print(f"  {activity:20s}: {acc:.2%}")

print("\n>>> LSTM - PER-CLASS ACCURACY")
y_test_lstm_labels = [activity_map[i+1] for i in y_test_lstm]
y_pred_lstm_labels = [activity_map[i+1] for i in y_pred_lstm]
for i, activity in enumerate(activity_labels_list):
    mask = np.array(y_test_lstm_labels) == activity
    acc = accuracy_score(np.array(y_test_lstm_labels)[mask], 
                        np.array(y_pred_lstm_labels)[mask]) if mask.sum() > 0 else 0
    print(f"  {activity:20s}: {acc:.2%}")

# ================= 10. SAVE MODELS =================

print("\n" + "=" * 60)
print("SAVING MODELS")
print("=" * 60)

model_lstm.save('lstm_har_model.h5')
model_cnn_lstm.save('cnn_lstm_har_model.h5')

import joblib
joblib.dump(lr_model, 'logistic_regression_model.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')

print("✅ All models saved:")
print("   - lstm_har_model.h5")
print("   - cnn_lstm_har_model.h5")
print("   - logistic_regression_model.pkl")
print("   - random_forest_model.pkl")

# ================= SUMMARY =================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

summary_data = {
    'Model': ['Logistic Regression', 'Random Forest', 'LSTM', 'CNN+LSTM'],
    'Type': ['ML', 'ML', 'DL', 'DL'],
    'Accuracy': [f"{acc_lr:.2%}", f"{acc_rf:.2%}", f"{acc_lstm:.2%}", f"{acc_cnn:.2%}"],
    'Best For': ['Baseline', 'Interpretability', 'Temporal', 'Complex Patterns']
}

summary_df = pd.DataFrame(summary_data)
print("\n", summary_df.to_string(index=False))

print("\n" + "=" * 60)
print("✅ Analysis Complete!")
print("=" * 60)
