import os
import mlflow
from mlflow.tracking import MlflowClient

def promote_model():
    # 1. Credentials
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    # 2. Correct Repository Details (Aapki repo)
    REPO_OWNER = "Suraj-Kumar09"
    REPO_NAME = "Capstone-Project"
    MODEL_NAME = "my_model"

    mlflow.set_tracking_uri(f'https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow')
    client = MlflowClient()

    # 3. SABSE LATEST Version uthayein (Staging stage par depend na rahein)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        print("No models found in Registry!")
        return
        
    # Version list mein se sabse bada number (latest) nikalein
    latest_version = max([int(v.version) for v in versions])
    print(f"Latest model version found: {latest_version}")

    # 4. Archive current Production models
    try:
        prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        for v in prod_versions:
            print(f"Archiving production version: {v.version}")
            client.transition_model_version_stage(MODEL_NAME, v.version, "Archived")
    except:
        print("No existing production model to archive.")

    # 5. Latest version ko Production mein Promote karein
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=str(latest_version),
        stage="Production"
    )
    print(f"Model version {latest_version} successfully promoted to Production.")

if __name__ == "__main__":
    promote_model()