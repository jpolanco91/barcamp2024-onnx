import torch
from torch import nn
from torch.utils.data import DataLoader

class DeepNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(17, 15),
            nn.Tanh(),
            nn.Linear(15,7),
            nn.Tanh(),
            nn.Linear(7,1)
        )
    def forward(self, x):
        return self.layers(x)

class WeightVariationDataset(torch.utils.data.Dataset):

    def __init__(self, X, y):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            self.X = torch.tensor(X.values)
            self.y = torch.tensor(y.values)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]
