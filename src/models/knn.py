
from sklearn.neighbors import KNeighborsClassifier
from src.models.base import BaseModel

class KNNModel(BaseModel):
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=5)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)