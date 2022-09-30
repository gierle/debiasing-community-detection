import torch
import torch.nn as nn

class FF(nn.Module):
    def __init__(self, num_input=10, num_classes=10, device='cpu'):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_input, int((num_input/4))),
            nn.ReLU(),
            nn.Linear(int((num_input/4)),  int((num_input/8))),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(int((num_input/8)), 100),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        ).to(device)
    
    def forward(self, x):
        return self.model(x)
