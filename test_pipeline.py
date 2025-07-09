import unittest
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense



class TestCyberSecurityPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup that runs once for all tests, like building models
        # Build a simple autoencoder for testing
        input_layer = Input(shape=(4,))
        encoded = Dense(2, activation='relu')(input_layer)
        decoded = Dense(4, activation='sigmoid')(encoded)
        cls.autoencoder = Model(input_layer, decoded)
        cls.encoder = Model(input_layer, encoded)

    def test_min_max_scaler(self):
        data = np.array([[1, 2], [2, 3], [3, 4]])
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(data)
        expected = np.array([[0., 0.], [0.5, 0.5], [1., 1.]])
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_autoencoder_encoding_shape(self):
        data = np.array([[1, 2, 3, 4]])
        encoded_data = self.encoder.predict(data)
        self.assertEqual(encoded_data.shape, (1, 2))

    def test_pca_transformation(self):
        data = np.random.rand(10, 5)
        pca = PCA(n_components=3)
        transformed = pca.fit_transform(data)
        self.assertEqual(transformed.shape, (10, 3))

    def test_random_forest_training_and_prediction(self):
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), 100)

    def test_mlp_training_and_prediction(self):
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=500)
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), 100)

if __name__ == '__main__':
    unittest.main()
