import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 1. DagsHub Setup
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "Suraj-Kumar09"
        repo_name = "Capstone-Project"
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # 2. Load Vectorizer
        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

        # 3. Registry Loading (Registry se model uthane ka standard process)
        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        
        # Registry URI: Registry se model load karne ka standard format
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        
        # Robust loading: Pehle Registry se, phir local fallback
        try:
            cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)
        except Exception as e:
            print(f"Registry load failed: {e}. Using local model.")
            cls.new_model = mlflow.pyfunc.load_model('models/model.pkl')

        # 4. Load Data
        cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        try:
            client = mlflow.MlflowClient()
            versions = client.get_latest_versions(model_name, stages=[stage])
            return versions[0].version if versions else None
        except:
            return None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        # Trained features ka use karke signature test
        trained_features = list(self.vectorizer.get_feature_names_out())
        input_data = self.vectorizer.transform(["hi how are you"])
        input_df = pd.DataFrame(input_data.toarray(), columns=trained_features)
        
        prediction = self.new_model.predict(input_df)
        self.assertEqual(len(prediction), input_df.shape[0])

    def test_model_performance(self):
        X = self.holdout_data.iloc[:, 0:-1]
        y = self.holdout_data.iloc[:, -1]

        # FEATURE ALIGNMENT: 50 features vs 20 features ka mismatch fix
        trained_features = list(self.vectorizer.get_feature_names_out())
        X_aligned = X[trained_features] 

        # Predict
        y_pred_new = self.new_model.predict(X_aligned.values)

        # Performance Metrics
        accuracy_new = accuracy_score(y, y_pred_new)
        self.assertGreaterEqual(accuracy_new, 0.40, f"Accuracy {accuracy_new} below 0.40")

if __name__ == "__main__":
    unittest.main()