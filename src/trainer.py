import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

def train_model(model, X_train, y_train, epochs=30, batch_size=32, model_name="rnn_model"):
    os.makedirs("models/", exist_ok=True)

    checkpoint = ModelCheckpoint(
        f"models/{model_name}.keras", monitor="loss", save_best_only=True, verbose=1
    )
    early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    return history