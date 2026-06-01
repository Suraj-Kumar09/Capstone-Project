import numpy as np
import pandas as pd
import pickle
import json
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
from src.logger import logging

# Dynamic DagsHub & MLflow Configuration
mlflow.set_tracking_uri('https://dagshub.com/Suraj-Kumar09/Capstone-Project.mlflow')

if os.getenv("GITHUB_ACTIONS"):
    dagshub_token = os.getenv("ATLAS")
    if not dagshub_token:
        raise EnvironmentError("ATLAS environment variable is not set")
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    dagshub.init(repo_owner='Suraj-Kumar09', repo_name='Capstone-Project', mlflow=True)
else:
    dagshub.init(repo_owner='Suraj-Kumar09', repo_name='Capstone-Project', mlflow=True)

def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file: 
            model = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except Exception as e:
        logging.error('Error loading model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except Exception as e:
        logging.error('Error loading data: %s', e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
        metrics_dict = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        logging.info('Metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error('Error during evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(metrics, file, indent=4)
    logging.info('Metrics saved to %s', file_path)

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    model_info = {'run_id': run_id, 'model_path': model_path}
    with open(file_path, 'w') as file:
        json.dump(model_info, file, indent=4)

def main():
    mlflow.set_experiment("my-dvc-pipeline")
    with mlflow.start_run() as run:
        try:
            # Robust Path Handling
            base_dir = os.getcwd()
            clf = load_model(os.path.join(base_dir, 'models', 'model.pkl'))
            test_data = load_data(os.path.join(base_dir, 'data', 'processed', 'test_bow.csv'))
            
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(clf, X_test, y_test)
            
            save_metrics(metrics, os.path.join(base_dir, 'reports', 'metrics.json'))
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            if hasattr(clf, 'get_params'):
                mlflow.log_params(clf.get_params())
            
            mlflow.sklearn.log_model(clf, "model")
            save_model_info(run.info.run_id, "model", os.path.join(base_dir, 'reports', 'experiment_info.json'))
            mlflow.log_artifact(os.path.join(base_dir, 'reports', 'metrics.json'))

        except Exception as e:
            logging.error('Process failed: %s', e)
            raise

if __name__ == '__main__':
    main()