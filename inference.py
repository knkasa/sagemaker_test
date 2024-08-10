import joblib
import os
import numpy as np

def model_fn(model_dir):
    """Load the model from the model_dir directory."""
    model = joblib.load(os.path.join(model_dir, 'linear_regression_model.joblib'))
    return model

def input_fn(request_body, request_content_type):
    """Deserialize the request body into an appropriate format."""
    if request_content_type == 'text/csv':
        return np.array([float(x) for x in request_body.split(',')])
    raise ValueError(f'Unsupported content type: {request_content_type}')

def predict_fn(input_data, model):
    """Apply the model to the incoming request data."""
    return model.predict(input_data.reshape(-1, 1))

def output_fn(prediction, response_content_type):
    """Serialize the prediction into the response content type."""
    return str(prediction)
