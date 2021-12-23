import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBasic(nn.Module):
    def __init__(self):
        super(ConvBasic, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=2, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(6, 32, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.hidden1 = nn.Linear(54, 16)
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
       # out = self.relu2(self.conv2(out))
        #out = self.relu3(self.conv3(out))
        flat_out = out.view(out.size(0), -1)
        #out = self.relu4(self.hidden1(flat_out))
        return flat_out # 96

class GridObservationMLP(nn.Module):
    def __init__(self):
        super(GridObservationMLP, self).__init__()
        self.hidden1 = nn.Linear(25, 256)
        self.relu1 = nn.ReLU(inplace=False)
        self.hidden2 = nn.Linear(256, 512)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.relu1(self.hidden1(x.view(-1, 25)))
        out = self.relu2(self.hidden2(out))
        return out


# TODO: Possible remove belief and action GRU classes and use nn.GRU directly
# TODO: Check GRU num_layers

class beliefGRU(nn.Module):
    def __init__(self):
        super(beliefGRU, self).__init__()
        # Check input size
        self.gru1 = nn.GRU(516, 512, batch_first=True)

    def forward(self, x):
        out = self.gru1(x)
        return out


class actionGRU(nn.Module):
    def __init__(self):
        super(actionGRU, self).__init__()
        # Check input size
        self.gru1 = nn.GRU(4, 512, batch_first=True)

    def forward(self, x):
        out = self.gru1(x)
        return out


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Check input size
        self.hidden1 = nn.Linear(1024, 200)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(200, 50)
        self.relu2 = nn.ReLU()
        self.hidden3 = nn.Linear(50, 1)

    def forward(self, x):
        out = self.relu1(self.hidden1(x))
        out = self.relu2(self.hidden2(out))
        out = self.hidden3(out)
        return out



class evalMLP(nn.Module):
    def __init__(self, grid_dims):
        super(evalMLP, self).__init__()
        self.x_size, self.y_size = grid_dims
        # TODO: init input size - b_t + grid_size[0]*grid_size[1]
        # TODO: change size to accomodate orientation
        self.hidden1 = nn.Linear(512,  300)
        self.relu1 = nn.ReLU()
        self.hidden3 = nn.Linear(300, 4)
        self.hidden4 = nn.Linear(300, self.x_size * self.y_size)

    def forward(self, x):
        out = self.relu1(self.hidden1(x))
        out_1 = self.hidden3(out)
        out_2 = self.hidden4(out)
        return out_2, out_1