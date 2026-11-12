import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP (nn.Module):
    def __init__(self, vocab_size=23, window_size=3, d=16, h=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.embedding  = nn.Embedding(vocab_size, d)
        self.fc1  = nn.Linear(window_size * d, h)
        self.fc2  = nn.Linear(h, vocab_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = F.pad(x, pad=(2, 0), value=0)
        x= x.unfold(dimension=1, size=3, step=1)
        x = self.embedding(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x) 
        return x