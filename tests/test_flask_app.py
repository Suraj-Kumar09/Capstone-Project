import unittest
import os
from flask_app.app import app

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Flask testing client
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        # Title ko exact match karein (dashboard add kiya hai)
        self.assertIn(b'<title>Sentiment Analysis Dashboard</title>', response.data)

    def test_predict_page(self):
        # POST request with sample text
        response = self.client.post('/predict', data={'text': "I love this!"})
        
        # Agar yahan 500 error aaye, to iska matlab app model load nahi kar pa rahi
        self.assertEqual(response.status_code, 200, f"Predict page failed with code {response.status_code}. Response: {response.data}")
        
        self.assertTrue(
            b'Positive' in response.data or b'Negative' in response.data,
            "Response should contain either 'Positive' or 'Negative'"
        )

if __name__ == '__main__':
    unittest.main()