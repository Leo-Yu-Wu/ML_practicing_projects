import csv
from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataSet(Dataset):
    def __init__(self, csv_file):
        self.data = []

        # load csv file
        with open(csv_file, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)    # skip first line
            for row in reader: # for each line
                x, y, z = map(int, row[:3]) #   get x,y,z value (first 3 col)
                label = 0 if row[-1] == 'A' else 1 #get label (0 for A, 1 for B)
                self.data.append([(x, y, z), label])  # append data

    def __getitem__(self, index):
        x_data, y_data = self.data[index]
        x = torch.tensor(x_data, dtype=torch.float)
        y = torch.tensor(y_data, dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.data)

# load dataset
train_file = 'data/train.csv'
test_file = 'data/test.csv'
batch_size = 32
train_data, test_data = CustomDataSet(train_file), CustomDataSet(test_file)

def get_data_loaders(batch_size:int=32):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader