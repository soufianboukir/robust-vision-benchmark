import torch.nn as nn


# implementation of MLP with 5 fully connected layers + dropout
class MLP5(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(32*32*3, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),  
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)