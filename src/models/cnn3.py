import torch.nn as nn

class CNN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # conv block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # producing 32 feature maps, 3 for nm of channels
            nn.Relu(),
            nn.MaxPool2d(2,2),

            # conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Relu(),
            nn.MaxPool2d(2,2),

            # conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Relu(),
            nn.MaxPool2d(2,2),

            # fully connected layers
            nn.Flatten(),
            nn.Conv2d(128 * 4 * 4, 256),
            nn.Relu(),
            nn.Conv2d(256,10),
        )

    def forware(self, x):
        return self.net(x)