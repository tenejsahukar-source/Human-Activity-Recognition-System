from app.ml.preprocess import preprocess
import numpy as np
import os

def load_single_sample(folder_path, index=0):

    signal_files = [
        "body_acc_x_test.txt",
        "body_acc_y_test.txt",
        "body_acc_z_test.txt",
        
        "body_gyro_x_test.txt",
        "body_gyro_y_test.txt",
        "body_gyro_z_test.txt",
        
        "total_acc_x_test.txt",
        "total_acc_y_test.txt",
        "total_acc_z_test.txt"
    ]

    signals = []

    for file in signal_files:
        data = np.loadtxt(os.path.join(folder_path, file))
        signals.append(data[index])  # pick same row

    # shape → (9, 128) → transpose → (128, 9)
    sample = np.array(signals).T

    return sample