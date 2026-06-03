import os
import mlflow
from mlflow.tracking import MlflowClient

def promote_model():
    # 1. Credentials Setup
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    # 2. Define repository details
    dagshub_url = "https://dagshub.com"
    repo_owner = "Suraj-Kumar09"
    repo_name = "Capstone-Project"

    # 3. Set up MLflow tracking URI using variables
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
    
    client = MlflowClient()
    model_name = "my_model"

    # 4. Get the latest version in Staging
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    
    if not staging_versions:
        print("No model version found in 'Staging'. Promotion aborted.")
        return

    latest_version_staging = staging_versions[0].version
    print(f"Latest version in Staging: {latest_version_staging}")

    # 5. Archive existing Production models
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        print(f"Archiving production version: {version.version}")
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # 6. Promote the new model to Production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"Successfully promoted model version {latest_version_staging} to Production")

if __name__ == "__main__":
    promote_model()