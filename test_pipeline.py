from src.data_loader import fetch_stock_data
from src.preprocessing import load_and_clean, scale_data, create_sequences
from src.model_rnn import create_rnn
from src.trainer import train_model
from src.predictor import predict_next
import tensorflow as tf
# Check TensorFlow and GPU availability
print("✅ TensorFlow version:", tf.__version__)
print("🧠 GPU devices available:", tf.config.list_physical_devices('GPU'))

# 1. Download data
df = fetch_stock_data("BBVA.MC")

# 2. Preprocess
clean_df = df[['Close']].dropna()
scaled_data, scaler = scale_data(clean_df)
X, y = create_sequences(scaled_data, lookback=30)

# 3. Build model
model = create_rnn(input_shape=(X.shape[1], X.shape[2]))

# 4. Train model
train_model(model, X, y, epochs=5)

# 5. Predict next day
prediction = predict_next("models/rnn_model.keras", scaled_data)
print(f"Predicted scaled close: {prediction}")