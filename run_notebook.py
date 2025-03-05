import boto3
import sagemaker
from sagemaker.processing import ScriptProcessor

# Note: If you setup CLI, you do not need the credential.
session = boto3.Session(
     aws_access_key_id='xxx',
     aws_secret_access_key='yyy',
     region_name='ap-northeast-1'
    ) 

client = session.client('sagemaker')

# Define SageMaker session and role. (Create the role if needed.)
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::your-account-id:role/service-role/AmazonSageMaker-ExecutionRole"

# Choose your image.
image_url = "7777777777.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-ubuntu20.04-sagemaker", 

# Define the processor
processor = ScriptProcessor(
    image_uri=image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    sagemaker_session=sagemaker_session
    )

# Run the notebook script on SageMaker
processor.run(
    code="your_notebook.py",
    inputs=[],
    outputs=[]
    )

print("Notebook executed on SageMaker!")
