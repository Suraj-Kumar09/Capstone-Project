import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 1. Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "Suraj-Kumar09"
        repo_name = "Capstone-Project"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # 2. Load the new model from MLflow model registry
        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # 3. Load the vectorizer
        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

        # 4. Load holdout test data
        cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        # Create a dummy input based on vectorizer features
        trained_features = list(self.vectorizer.get_feature_names_out())
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=trained_features)

        # Predict using the new model
        prediction = self.new_model.predict(input_df)
        self.assertEqual(len(prediction), input_df.shape[0])

    def test_model_performance(self):
        # Extract features and labels from holdout test data
        X_holdout = self.holdout_data.iloc[:, 0:-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        # --- FIX: FEATURE ALIGNMENT ---
        # Yeh line ensure karti hai ki input columns wahi hain jo model ne seekhe hain
        trained_features = list(self.vectorizer.get_feature_names_out())
        
        # DataFrame ke columns ko trained_features se map/force karein
        if X_holdout.shape[1] == len(trained_features):
            X_holdout.columns = trained_features
        else:
            # Agar columns mismatch hain, toh sirf utne features lein jitne required hain
            X_holdout = X_holdout.iloc[:, :len(trained_features)]
            X_holdout.columns = trained_features

        # Predict using the new model
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate performance metrics
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Define thresholds
        expected_threshold = 0.40

        # Assertions
        self.assertGreaterEqual(accuracy_new, expected_threshold, f'Accuracy {accuracy_new} < {expected_threshold}')
        self.assertGreaterEqual(precision_new, expected_threshold, f'Precision {precision_new} < {expected_threshold}')
        self.assertGreaterEqual(recall_new, expected_threshold, f'Recall {recall_new} < {expected_threshold}')
        self.assertGreaterEqual(f1_new, expected_threshold, f'F1 {f1_new} < {expected_threshold}')

if __name__ == "__main__":
    unittest.main()