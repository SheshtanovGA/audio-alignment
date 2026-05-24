import os
import numpy as np
import pandas as pd

def safe_npy_load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Numpy file not found: {path}")
    return np.load(path)


def safe_npy_save(array, path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    np.save(path, array)


def safe_csv_read(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)

def safe_save_csv(df, path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    df.to_csv(path, index=False)
