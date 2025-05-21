# ----------------------------------------------------------------------------
# File: model.py
# Description: This file defines a neural network model using PyTorch for this classification problem
# Author: Leo Wu
# Date Created: 2025-05-21
# Last Modified: 2025-05-21
# Version: 1.0
# Dependencies: torch, torchvision, numpy
# Usage: Import this file to define the model and start training.
#         Example: `from model import create_model`
# ----------------------------------------------------------------------------
import torch.nn as nn

# model architecture
class Model(nn.Module):
    def __init__(self, input_size=3, hidden_1_size=4, hidden_2_size=8, output_size=2):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_1_size)
        self.linear2 = nn.Linear(hidden_1_size, hidden_2_size)
        self.out = nn.Linear(hidden_2_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.relu(self.linear1(input))
        x = self.relu(self.linear2(x))
        x = self.out(x)
        return x