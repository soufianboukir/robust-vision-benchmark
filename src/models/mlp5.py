import torch.nn as nn


# implementation of MLP with 5 fully connected layers + dropout
class MLP5(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(32*32*3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),  
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)