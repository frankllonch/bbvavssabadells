import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

# Print available devices
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

def create_rnn(input_shape):
    """Create and compile a simple RNN model."""
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        model = Sequential([
            SimpleRNN(128, activation='tanh', return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            SimpleRNN(128, activation='tanh'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
    return model