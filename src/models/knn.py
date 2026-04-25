from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from src.models.base import BaseModel


class KNNModel(BaseModel):
    def __init__(self, n_components=500, n_neighbors=5):
        self.pca = PCA(n_components=n_components)
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X, y):
        X_reduced = self.pca.fit_transform(X)
        self.model.fit(X_reduced, y)

    def predict(self, X):
        X_reduced = self.pca.transform(X)
        return self.model.predict(X_reduced)