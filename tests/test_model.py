import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from mlflow.tracking import MlflowClient

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 1. Setup credentials
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")
        
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # Define repository details
        dagshub_url = "https://dagshub.com"
        repo_owner = "Suraj-Kumar09"
        repo_name = "Capstone-Project"

        # 2. Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # 3. Load Model: Production priority with Local Fallback
        cls.new_model_name = "my_model"
        try:
            client = MlflowClient()
            # Production stage se best model uthayenge
            versions = client.get_latest_versions(cls.new_model_name, stages=["Production"])
            if not versions:
                raise Exception("No Production model found")
            
            cls.new_model_version = versions[0].version
            print(f"Loading Production model version: {cls.new_model_version}")
            cls.new_model = mlflow.pyfunc.load_model(f'models:/{cls.new_model_name}/{cls.new_model_version}')
        except Exception as e:
            print(f"Registry load failed, falling back to local: {e}")
            cls.new_model = mlflow.pyfunc.load_model(os.path.abspath('models/model.pkl'))

        # 4. Load Vectorizer & Data
        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
        cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')

    @staticmethod
    def get_latest_model_version(model_name, stage="Production"):
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])
        return versions[0].version if versions else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        trained_features = list(self.vectorizer.get_feature_names_out())
        input_df = pd.DataFrame(self.vectorizer.transform(["hi how are you"]).toarray(), columns=trained_features)
        prediction = self.new_model.predict(input_df)
        self.assertEqual(len(prediction), input_df.shape[0])

    def test_model_performance(self):
        # 1. Prepare data
        X = self.holdout_data.iloc[:, 0:-1]
        y = self.holdout_data.iloc[:, -1]

        # 2. FEATURE ALIGNMENT:
        # Train features ke naam extract karein
        trained_features = list(self.vectorizer.get_feature_names_out())
        
        # Mismatch handle karne ke liye alignment
        if X.shape[1] > len(trained_features):
            X = X.iloc[:, :len(trained_features)]
        
        # DataFrame columns ko force align karein
        X.columns = trained_features

        # 3. Predict
        y_pred = self.new_model.predict(X)

        # 4. Metrics & Assertions
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        threshold = 0.40
        self.assertGreaterEqual(accuracy, threshold, f"Accuracy {accuracy} < {threshold}")
        self.assertGreaterEqual(precision, threshold, f"Precision {precision} < {threshold}")
        self.assertGreaterEqual(recall, threshold, f"Recall {recall} < {threshold}")
        self.assertGreaterEqual(f1, threshold, f"F1 {f1} < {threshold}")

if __name__ == "__main__":
    unittest.main()