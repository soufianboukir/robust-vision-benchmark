from sklearn.linear_model import LogisticRegression
from src.models.base import BaseModel
from sklearn.decomposition import PCA

class LogisticModel(BaseModel):
    def __init__(self, n_components=20):
        self.pca = PCA(n_components=n_components)
        self.model = LogisticRegression(
            solver="saga",
            max_iter=3000
        )

    def fit(self, X, y):
        X_reduced = self.pca.fit_transform(X)
        self.model.fit(X_reduced, y)

    def predict(self, X):
        X_reduced = self.pca.transform(X)
        return self.model.predict(X_reduced)