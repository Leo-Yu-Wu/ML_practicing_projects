from data import get_data_loaders
from model import Model
from train import train_model
from validate import validate_model

import torch
import torch.optim as optim
import torch.nn as nn

def main():
    batch_size = 32
    learning_rate = 0.001
    epoch = 200
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train model
    train_model(model, device, train_loader, optimizer, criterion, num_epoch=epoch)
    torch.save(model.state_dict(), 'models/model.pth')

    # validate model
    model.load_state_dict(torch.load('models/model.pth', weights_only=True))
    validate_model(model, test_loader, device)

if __name__ == '__main__':
    main()