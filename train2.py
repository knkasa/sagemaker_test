# Instrunctin to use train.py with your own modules. Specify this in sagemaker estimater.
# You should probably use docker image if you have too many modules.

# Project structure:
# my_sagemaker_project/
# ├── source/
# │   ├── train.py
# │   ├── model.py
# │   ├── data_processor.py
# │   ├── utils.py
# │   └── requirements.txt
# └── run_training.py

# File: source/model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_lstm_model(n_features, lstm_units):
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, input_shape=(None, n_features)),
        Dropout(0.2),
        LSTM(units=lstm_units//2, return_sequences=False),
        Dropout(0.2),
        Dense(units=32, activation='relu'),
        Dense(units=1)
    ])
    return model

# File: source/data_processor.py
import numpy as np

def preprocess_data(data):
    # Your data preprocessing logic
    return processed_data

def create_sequences(data, seq_length):
    # Your sequence creation logic
    return sequences

# File: source/utils.py
def calculate_metrics(y_true, y_pred):
    # Your metrics calculation logic
    return metrics

# File: source/train.py
import os
import argparse
from model import create_lstm_model
from data_processor import preprocess_data, create_sequences
from utils import calculate_metrics

def train():
    # Your training logic here
    pass

if __name__ == '__main__':
    train()

# File: source/requirements.txt
pandas==1.5.3
scikit-learn==1.2.2

#=================== Within notebook ==================================================
import sagemaker
from sagemaker.tensorflow import TensorFlow

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()

# Define hyperparameters
hyperparameters = {
    'epochs': 10,
    'batch-size': 32,
    'learning-rate': 0.001,
    'lstm-units': 50
}

# Create TensorFlow estimator with source_dir
estimator = TensorFlow(
    entry_point='train.py',  # Main training script
    source_dir='source/',    # Directory containing all your Python modules
    role=role,
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    framework_version='2.12.0',
    py_version='py39',
    hyperparameters=hyperparameters,
    output_path='s3://your-bucket/output',
    requirements_file='source/requirements.txt'  # Optional: for additional dependencies
)

# Start training
estimator.fit({
    'train': 's3://your-bucket/train',
    'test': 's3://your-bucket/test'
})