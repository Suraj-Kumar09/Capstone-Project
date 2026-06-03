import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score
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
        mlflow.set_tracking_uri('https://dagshub.com/Suraj-Kumar09/Capstone-Project.mlflow')

        # 2. Load Vectorizer
        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

        # 3. Robust Model Loading
        cls.new_model_name = "my_model"
        try:
            cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
            cls.new_model = mlflow.pyfunc.load_model(f'models:/{cls.new_model_name}/{cls.new_model_version}')
        except:
            cls.new_model = mlflow.pyfunc.load_model('models/model.pkl')

        # 4. Load holdout test data (CSV mein 'text' aur 'target' column hona chahiye)
        cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        try:
            client = mlflow.MlflowClient()
            versions = client.get_latest_versions(model_name, stages=[stage])
            return versions[0].version if versions else None
        except:
            return None

    def test_model_performance(self):
        # FIX: Agar CSV mein columns match nahi ho rahe, 
        # toh hum text ko lekar vectorizer se dobara transform karenge
        # (Assuming your test_bow.csv has a 'text' column)
        
        if 'text' in self.holdout_data.columns:
            X = self.vectorizer.transform(self.holdout_data['text'])
            y = self.holdout_data['target']
        else:
            # Fallback agar CSV mein sirf features hain
            X = self.holdout_data.iloc[:, 0:-1]
            y = self.holdout_data.iloc[:, -1]
            trained_features = list(self.vectorizer.get_feature_names_out())
            # Column mapping check
            if all(col in X.columns for col in trained_features):
                X = X[trained_features]

        # Predict
        y_pred = self.new_model.predict(X)

        # Performance Check
        accuracy = accuracy_score(y, y_pred)
        self.assertGreaterEqual(accuracy, 0.40, f"Accuracy {accuracy} below 0.40")

if __name__ == "__main__":
    unittest.main()