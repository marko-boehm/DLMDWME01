from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
import numpy as np

class Rectangle:
    def __init__(self, height):
        self.height = height

# GMM classifier
class GMMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=1, covariance_type='full', init_params='kmeans', random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.init_params = init_params
        self.gmm_0 = None
        self.gmm_1 = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X0 = X[y == 0]
        X1 = X[y == 1]
        
        self.gmm_0 = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type, random_state=self.random_state)
        self.gmm_1 = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type, random_state=self.random_state)
        
        self.gmm_0.fit(X0)
        self.gmm_1.fit(X1)

        return self

    def predict_proba(self, X):
        proba_0 = self.gmm_0.score_samples(X)
        proba_1 = self.gmm_1.score_samples(X)
        proba = np.exp(np.vstack([proba_0, proba_1]).T)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)