import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
import math
import matplotlib.pyplot as plt
import logging
import os

# Milvus imports
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# Enable mixed precision for performance improvements (requires GPU/TPU)
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='model_training.log', filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# Step 1: Connect to Milvus
def connect_to_milvus():
    try:
        connections.connect(host='localhost', port='19530')
        logging.info("Connected to Milvus")
    except Exception as e:
        logging.error(f"Failed to connect to Milvus: {e}")
        return
    logging.info("Connected to Milvus")

# Step 2: Create Milvus Collection (Schema for feature vectors)
def create_milvus_collection(collection_name, dim):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="feature_vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description="Stock feature vector collection")
    collection = Collection(name=collection_name, schema=schema)
    logging.info(f"Milvus collection '{collection_name}' created with dimension {dim}.")
    return collection

# Step 3: Insert feature vectors into Milvus
def insert_vectors(collection, feature_vectors):
    collection.insert([feature_vectors])
    logging.info(f"Inserted {len(feature_vectors)} vectors into Milvus collection '{collection.name}'.")

# Step 4: Create an index for the collection
def create_index(collection):
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024},
    }
    collection.create_index(field_name="feature_vector", index_params=index_params)
    logging.info(f"Index created for collection '{collection.name}'.")

# Step 5: Load the Milvus collection into memory
def load_collection(collection):
    collection.load()
    logging.info(f"Collection '{collection.name}' loaded into memory.")

# Step 6: Search similar vectors in Milvus
def search_similar_vectors(collection, query_vector, top_k=5):
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_vector],
        anns_field="feature_vector",
        param=search_params,
        limit=top_k,
    )
    logging.info(f"Found {len(results)} similar vectors.")
    return results

def calculate_rsi(data, window=14):
    try:
        delta = data['close'].diff(1)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=window).mean()
        avg_loss = pd.Series(loss).rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        logging.error(f"Error calculating RSI: {e}")
        raise

def calculate_bollinger_bands(data, window=20, num_sd=2):
    try:
        sma = data['close'].rolling(window=window).mean()
        std_dev = data['close'].rolling(window=window).std()
        upper_band = sma + (num_sd * std_dev)
        lower_band = sma - (num_sd * std_dev)
        return upper_band, lower_band
    except Exception as e:
        logging.error(f"Error calculating Bollinger Bands: {e}")
        raise

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    try:
        short_ema = data['close'].ewm(span=short_window, adjust=False).mean()
        long_ema = data['close'].ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        return macd, signal
    except Exception as e:
        logging.error(f"Error calculating MACD: {e}")
        raise

def scale_in_batches(data, scaler, batch_size=1000):
    try:
        scaled_data = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            scaled_batch = scaler.fit_transform(batch)
            scaled_data.append(scaled_batch)
        return np.vstack(scaled_data)
    except Exception as e:
        logging.error(f"Error scaling data in batches: {e}")
        raise

def create_sequences(data, time_steps=50, augment=False):
    try:
        X, y = [], []
        for i in range(len(data) - time_steps):
            sequence = data[i:i + time_steps, :]
            if augment:
                noise = np.random.normal(0, 0.005, sequence.shape)
                sequence += noise
            X.append(sequence)
            y.append(data[i + time_steps, 3])  # Predicting 'close' price
        return np.array(X), np.array(y)
    except Exception as e:
        logging.error(f"Error creating sequences: {e}")
        raise

def build_lstm_dnn_model(input_shape):
    try:
        lstm_input = Input(shape=(input_shape[1], input_shape[2]))

        # LSTM Layer with L2 regularization and Dropout
        lstm_out = Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=l2(1e-4)))(lstm_input)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Dropout(0.4)(lstm_out)

        # Dense Layers with L2 regularization and increased Dropout
        dense_out = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(lstm_out)
        dense_out = BatchNormalization()(dense_out)
        dense_out = Dropout(0.3)(dense_out)

        dense_out = Dense(32, activation='relu', kernel_regularizer=l2(1e-4))(dense_out)
        dense_out = BatchNormalization()(dense_out)
        dense_out = Dropout(0.3)(dense_out)

        # Output Layer
        output = Dense(1, dtype='float32')(dense_out)

        # Compile the Model
        model = Model(inputs=lstm_input, outputs=output)
        optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-5, clipvalue=1.0)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mean_squared_error'])

        logging.info("Model successfully built and compiled.")
        return model
    except Exception as e:
        logging.error(f"Error building the model: {e}")
        raise

def calculate_baseline_performance(model, X_test, y_test_actual, scaler):
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(
        np.concatenate([X_test[:, -1, :-1], predicted_prices.reshape(-1, 1)], axis=1)
    )[:, -1]
    baseline_rmse = math.sqrt(mean_squared_error(y_test_actual, predicted_prices))
    return baseline_rmse

def permutation_importance(model, X_test, y_test_actual, features, scaler):
    baseline_rmse = calculate_baseline_performance(model, X_test, y_test_actual, scaler)
    importances = {}

    for i in range(X_test.shape[2]):  # Iterate over each feature
        X_test_permuted = np.copy(X_test)
        np.random.shuffle(X_test_permuted[:, :, i])

        permuted_predicted_prices = model.predict(X_test_permuted)
        permuted_predicted_prices = scaler.inverse_transform(
            np.concatenate([X_test[:, -1, :-1], permuted_predicted_prices.reshape(-1, 1)], axis=1)
        )[:, -1]

        permuted_rmse = math.sqrt(mean_squared_error(y_test_actual, permuted_predicted_prices))

        importances[features[i]] = permuted_rmse - baseline_rmse

    importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    return importances

def directional_accuracy(y_actual, y_pred):
    actual_diff = np.diff(y_actual)  # Difference between consecutive actual values
    predicted_diff = np.diff(y_pred)  # Difference between consecutive predicted values
    
    correct_directions = np.sum(np.sign(actual_diff) == np.sign(predicted_diff))  # Count matching directions
    total_directions = len(actual_diff)
    
    return correct_directions / total_directions * 100  # Return percentage of correct directions

def main():
    try:
        # Step 1: Connect to Milvus
        connect_to_milvus()

        # Step 2: Load and Preprocess the Dataset
        if not os.path.exists("./Nifty 50 minute.csv"):
            logging.error("Dataset file not found!")
            return
        
        nifty_data = pd.read_csv("./Nifty 50 minute.csv", encoding='ISO-8859-1')
        nifty_data['date'] = pd.to_datetime(nifty_data['date'])
        nifty_data = nifty_data.sort_values('date')

        # Feature Engineering
        logging.info("Starting feature engineering...")
        nifty_data['SMA_20'] = nifty_data['close'].rolling(window=20).mean()
        nifty_data['EMA_20'] = nifty_data['close'].ewm(span=20, adjust=False).mean()
        nifty_data['log_return'] = np.log(nifty_data['close'] / nifty_data['close'].shift(1))
        nifty_data['volatility'] = nifty_data['log_return'].rolling(window=10).std()
        nifty_data['RSI'] = calculate_rsi(nifty_data)
        nifty_data['upper_band'], nifty_data['lower_band'] = calculate_bollinger_bands(nifty_data)
        nifty_data['MACD'], nifty_data['MACD_signal'] = calculate_macd(nifty_data)

        nifty_data.dropna(inplace=True)
        logging.info("Feature engineering completed.")

        # Step 3: Feature Selection
        features = ['open', 'high', 'low', 'close', 'volume', 'SMA_20', 'EMA_20', 'log_return', 'volatility', 'RSI', 'upper_band', 'lower_band', 'MACD', 'MACD_signal']

        # Step 4: Normalize the data using Batch-Wise Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(nifty_data[features])
        nifty_scaled = scale_in_batches(nifty_data[features], scaler, batch_size=1000)
        logging.info("Data scaled successfully.")

        # Step 5: Insert vectors into Milvus
        collection = create_milvus_collection('nifty_feature_vectors', dim=len(features))
        insert_vectors(collection, nifty_scaled)

        # Step 6: Create an index for the collection
        create_index(collection)

        # Step 7: Load the collection into memory before searching
        load_collection(collection)

        # Example query for similar vectors (optional):
        example_vector = nifty_scaled[0]  # Use one vector to search
        search_similar_vectors(collection, example_vector)

        # Step 8: Create Sequences
        X, y = create_sequences(nifty_scaled, time_steps=50, augment=True)
        logging.info("Sequences created successfully.")

        # Step 9: Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Step 10: Prepare the TensorFlow Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.cache().shuffle(buffer_size=1000).batch(64).prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(64).prefetch(tf.data.experimental.AUTOTUNE)

        # Step 11: Build the Model
        input_shape = X_train.shape  # Get the shape of X_train
        lstm_model = build_lstm_dnn_model(input_shape)

        # Step 12: Train the Model
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        lr_scheduler = LearningRateScheduler(lambda epoch, lr: lr * 0.9 if epoch >= 3 else lr * 1.1)
        lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=1e-7)

        logging.info("Training the model...")
        history = lstm_model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=[early_stopping, lr_scheduler, lr_reduction])
        logging.info("Model training completed.")

        # Step 13: Plot the Learning Curve
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss', color='blue')
        plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
        plt.title('Learning Curve: Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig('learning_curve.png')
        logging.info("Learning curve saved as 'learning_curve.png'.")

        # Step 14: Evaluate the Model on Test Data
        time_steps = 50  # Define time_steps here
        X_test, y_test = create_sequences(nifty_scaled[-10000:], time_steps)
        predicted_prices = lstm_model.predict(X_test)


        # Inverse transform to get actual prices
        predicted_prices = scaler.inverse_transform(
            np.hstack([np.zeros((predicted_prices.shape[0], len(features)-1)), predicted_prices])
        )[:, -1]

        # Similarly, inverse transform the actual 'close' price for comparison
        y_test_actual = scaler.inverse_transform(
            np.hstack([np.zeros((y_test.shape[0], len(features)-1)), y_test.reshape(-1, 1)])
        )[:, -1]

        # Step 15: Permutation Feature Importance
        importances = permutation_importance(lstm_model, X_test, y_test_actual, features, scaler)
        for feature, importance in importances:
            print(f'Feature: {feature}, Importance: {importance}')

        # Step 16: Calculate Metrics
        directional_acc = directional_accuracy(y_test_actual, predicted_prices)

        mape = mean_absolute_percentage_error(y_test_actual, predicted_prices)
        rmse = math.sqrt(mean_squared_error(y_test_actual, predicted_prices))
        mae = mean_absolute_error(y_test_actual, predicted_prices)
        r2 = r2_score(y_test_actual, predicted_prices)

        print(f"Test MAPE: {mape}")
        print(f"Test RMSE: {rmse}")
        print(f"Test MAE: {mae}")
        print(f"R2 Score: {r2}")
        print(f"Directional Accuracy: {directional_acc:.2f}%")

        # Step 17: Plot Actual vs Predicted Prices
        plt.figure(figsize=(15, 8))
        plt.plot(y_test_actual, label='Actual Prices', color='b')
        plt.plot(predicted_prices, label='Predicted Prices', color='r')
        plt.title('Bidirectional LSTM + DNN: Actual vs Predicted Prices')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

        # Step 18: Save the Model
        lstm_model.save('bidirectional_lstm_dnn_model_with_regularization_and_dropout.h5')
        logging.info("Model saved as 'bidirectional_lstm_dnn_model_with_regularization_and_dropout.h5'.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
