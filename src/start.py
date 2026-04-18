from config.config import Config

from models.mlp3 import MLP3
from models.mlp5 import MLP5
from models.cnn3 import CNN3
from models.cnn5 import CNN5
from models.cnn7 import CNN7
from models.mlp3 import MLP3
from models.resnet18 import ResNet18
from models.resnet50 import ResNet50


def get_model(model):
    if model == 'mlp3':
        return 