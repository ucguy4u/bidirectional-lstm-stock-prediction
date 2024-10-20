
# Bidirectional LSTM + DNN Stock Price Prediction with Milvus Integration

This project predicts stock prices using a Bidirectional LSTM and DNN model with features engineered from historical stock data. The feature vectors are stored and queried using Milvus, an open-source vector database optimized for high-dimensional data.

## Features

- Bidirectional LSTM combined with Dense layers
- Dropout and L2 Regularization for better generalization
- Mixed precision training for performance improvement
- Feature engineering (SMA, EMA, RSI, MACD, Bollinger Bands)
- Data storage and query in Milvus Vector Database
- Model training with TensorFlow Data API (Caching and Prefetching)
- Callbacks for learning rate scheduling and early stopping
- Performance metrics: MAPE, RMSE, MAE, R2 Score, Directional Accuracy
- Permutation Feature Importance analysis
- Visualization of learning curves and predicted vs actual prices

## Setup

### Prerequisites

- Docker and Docker Compose installed
- Python 3.8+
- Required Python packages (see `requirements.txt`)

### Install Python Dependencies

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/stock-price-prediction-lstm-dnn-milvus.git
    cd stock-price-prediction-lstm-dnn-milvus
    ```

2. Install required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Milvus Setup

Milvus is used as a vector database to store and query feature vectors. Follow these steps to set up and run Milvus.

1. Download the Milvus standalone Docker Compose file:

    ```bash
    wget https://github.com/milvus-io/milvus/releases/download/v2.0.2/milvus-standalone-docker-compose.yml -O docker-compose.yml
    ```

2. Start Milvus:

    ```bash
    sudo docker-compose up -d
    ```

    If you are using Docker Compose V2, run:

    ```bash
    sudo docker compose up -d
    ```

3. Verify that Milvus is running:

    ```bash
    sudo docker-compose ps
    ```

    You should see the following containers up and running:

    ```
    Name                     Command                  State                          Ports
    -----------------------------------------------------------------------------------------------------------
    milvus-etcd         etcd -listen-peer-urls=htt ...   Up (healthy)   2379/tcp, 2380/tcp
    milvus-minio        /usr/bin/docker-entrypoint ...   Up (healthy)   9000/tcp
    milvus-standalone   /tini -- milvus run standalone   Up             0.0.0.0:19530->19530/tcp,:::19530->19530/tcp
    ```

4. To stop Milvus:

    ```bash
    sudo docker-compose down
    ```

5. To delete the stored data after stopping Milvus:

    ```bash
    sudo rm -rf volumes
    ```

### Running the Model

1. Ensure that Milvus is running as explained in the Milvus setup section.

2. Run the Python script:

    ```bash
    python3 main.py
    ```

3. During the run, the script will:

    - Load and preprocess stock data.
    - Store feature vectors in Milvus.
    - Query Milvus for similar vectors (optional).
    - Train a Bidirectional LSTM + DNN model using TensorFlow.
    - Save the model and plot the learning curve.

### Data

The dataset used is stock price data, which includes features such as Open, High, Low, Close, and Volume. The project also adds technical indicators such as:

- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- Bollinger Bands
- MACD

You can use your own stock price dataset or download a sample dataset.

### Saving and Loading the Model

The trained model is saved as `bidirectional_lstm_dnn_model_with_regularization_and_dropout.h5`. You can load the model using:

```python
from tensorflow.keras.models import load_model
model = load_model('bidirectional_lstm_dnn_model_with_regularization_and_dropout.h5')
```

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
