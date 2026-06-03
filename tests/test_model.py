import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "Suraj-Kumar09"
        repo_name = "Capstone-Project"
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load the vectorizer
        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

        # Load model with Fallback Logic
        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        
        if cls.new_model_version:
            cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
            cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)
        else:
            cls.new_model = mlflow.pyfunc.load_model('models/model.pkl')

        # Load holdout test data
        cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        try:
            client = mlflow.MlflowClient()
            latest_version = client.get_latest_versions(model_name, stages=[stage])
            return latest_version[0].version if latest_version else None
        except:
            return None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=self.vectorizer.get_feature_names_out())
        
        prediction = self.new_model.predict(input_df)
        self.assertEqual(len(prediction), input_df.shape[0])

    def test_model_performance(self):
        X_holdout = self.holdout_data.iloc[:, 0:-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        # FEATURE ALIGNMENT: 
        # Only use features that the model was trained on
        trained_features = list(self.vectorizer.get_feature_names_out())
        X_holdout_aligned = X_holdout[trained_features]

        # Predict using aligned features
        y_pred_new = self.new_model.predict(X_holdout_aligned.values)

        # Calculate metrics
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Assert performance
        self.assertGreaterEqual(accuracy_new, 0.40)
        self.assertGreaterEqual(precision_new, 0.40)
        self.assertGreaterEqual(recall_new, 0.40)
        self.assertGreaterEqual(f1_new, 0.40)

if __name__ == "__main__":
    unittest.main()