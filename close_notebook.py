import boto3

# Initialize SageMaker client
sagemaker_client = boto3.client("sagemaker", region_name="ap-northeast-1")

def stop_running_notebooks():
    # List all notebook instances
    response = sagemaker_client.list_notebook_instances(
        StatusEquals="InService"  # Only get running notebooks
    )

    notebooks = response.get("NotebookInstances", [])

    if not notebooks:
        print("No running SageMaker notebooks found.")
        return

    # Stop each running notebook
    for notebook in notebooks:
        notebook_name = notebook["NotebookInstanceName"]
        print(f"Stopping notebook: {notebook_name}...")
        sagemaker_client.stop_notebook_instance(NotebookInstanceName=notebook_name)

    print("All running SageMaker notebooks have been stopped.")

# Run the function
stop_running_notebooks()
