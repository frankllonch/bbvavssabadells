import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_clean(file_path: str):
    """Load and clean stock data."""
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df = df.dropna()
    return df

def scale_data(df: pd.DataFrame):
    """Scale features between 0 and 1."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

def create_sequences(data, lookback=30):
    """Create input/output sequences for RNN."""
    if isinstance(data, np.ndarray):
        arr = data
    else:
        arr = data.values  # convert DataFrame → NumPy

    X, y = [], []
    for i in range(len(arr) - lookback):
        X.append(arr[i:i + lookback])
        y.append(arr[i + lookback])
    return np.array(X), np.array(y)