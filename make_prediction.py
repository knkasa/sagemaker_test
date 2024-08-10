import boto3
import json
import pdb

# Note: If you setup CLI, you do not need the credential.
session = boto3.Session(
     aws_access_key_id='',
     aws_secret_access_key='+jhCDtVIBdda+o',
     region_name='ap-northeast-1'
    ) 

# Initialize boto3 SageMaker runtime client
runtime = session.client('sagemaker-runtime')

# Define the endpoint name
endpoint_name = 'sklearn-linear-regression-endpoint'

# Sample data for prediction (e.g., x=5)
payload = '5'

# Send the request to the SageMaker endpoint
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='text/csv',
    Body=payload
    )

# Parse the response
response_body = response['Body'].read().decode()
print(f'Predicted value for x=5: {response_body}')

