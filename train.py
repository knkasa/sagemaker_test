# Using custom training logic.  Save this as train.py. Then specify this file in sagemaker estimator.
'''
import sagemaker
from sagemaker.tensorflow import TensorFlow
import boto3

# Define S3 bucket and role
s3_bucket = "s3://your-bucket-name"
s3_prefix = "tensorflow-regression"
role = sagemaker.get_execution_role()

# Define the training script
training_script = "train.py"  # Your training script filename

# Set up the TensorFlow estimator
estimator = TensorFlow(
    entry_point=training_script,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version="2.9",
    py_version="py39",
    script_mode=True,
    output_path=f"{s3_bucket}/{s3_prefix}/output",
    hyperparameters={
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001
    },
    model_dir=f"{s3_bucket}/{s3_prefix}/model"
)

# Define S3 data paths
train_data = f"{s3_bucket}/{s3_prefix}/train"
validation_data = f"{s3_bucket}/{s3_prefix}/validation"

data_channels = {"train": train_data, "validation": validation_data}

# Train the model
estimator.fit(inputs=data_channels)'''

#============= train.py =================
import tensorflow as tf
import argparse
import os

# Parse command-line arguments for hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
args = parser.parse_args()

# Load dataset (Modify this based on your data format)
def load_data(data_dir):
    data_path = os.path.join(data_dir, 'data.csv')  # Assume CSV file
    dataset = tf.data.experimental.make_csv_dataset(
        data_path, batch_size=args.batch_size, label_name="target", num_epochs=1, shuffle=True
    )
    return dataset

train_dataset = load_data(args.train)
val_dataset = load_data(args.validation)

# Define the neural network model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model

# Create and train model
model = build_model()
model.fit(train_dataset, validation_data=val_dataset, epochs=args.epochs)

# Save model
model.save(args.model_dir)
