from sklearn.linear_model import LogisticRegression
from src.models.base import BaseModel

class LogisticModel(BaseModel):
    def __init__(self):
        self.model = LogisticRegression(
            solver="saga",
            max_iter=3000
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)