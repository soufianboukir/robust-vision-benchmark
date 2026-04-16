from sklearn.tree import DecisionTreeClassifier
from src.models.base import BaseModel

class DecisionTreeModel(BaseModel):
    def __init__(self):
        self.model = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)