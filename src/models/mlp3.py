import torch.nn as nn


# implementation of MLP with 3 fully connected layers
class MLP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*3, 512),
            nn.ReLU(),
            nn.Linear(512, 128), 
            nn.ReLU(),
            nn.Linear(128, 10) 
        )

    def forward(self, x):
        return self.net(x)