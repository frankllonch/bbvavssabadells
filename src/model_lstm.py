from keras.layers import LSTM

def train_lstm(X_train, y_train, X_val, y_val, lookback):
    model = Sequential([
        LSTM(64, activation='tanh', input_shape=(lookback, 1), return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=30, batch_size=32, verbose=1)
    model.save("models/lstm_model.keras")
    return model, history