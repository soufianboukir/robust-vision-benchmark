
from sklearn.svm import SVC
from src.models.base import BaseModel

class SVMModel(BaseModel):
    def __init__(self):
        self.model = SVC(kernel="rbf")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)