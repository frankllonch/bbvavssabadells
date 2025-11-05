import numpy as np
from tensorflow.keras.models import load_model

def predict_next(model_path, data, lookback=30):
    """Predict the next closing price based on recent data."""
    model = load_model(model_path)
    last_window = data[-lookback:]
    X = np.expand_dims(last_window, axis=0)
    prediction = model.predict(X)
    return prediction[0][0]