import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch import optim
from io import StringIO

features = 3
input = 2 * features
LearningRate = 0.01






class net(nn.Module):
    def __init__(self, ):
        super(net, self).__init__()
        self.features = input
        self.fc0 = nn.BatchNorm1d(input)
        self.fc1 = nn.Linear(input, input * 2)
        self.fc2 = nn.BatchNorm1d(input * 2)
        self.fc3 = nn.Linear(input * 2, 2)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.fc0(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, optimizer, data, label):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, label)
    loss.backward()
    optimizer.step()

def prediction(model,data):
    model.eval()
    output = model(data)
    pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
    return pred


def learn(flights):
    _net = net()
    optimizer = optim.Adam(_net.parameters(), lr=LearningRate)

    print("here!")

# hyper : optimazer + Learning Rate
# test 1: 5 random flights vs. 5 best of each features
# test 2: auto agent - always choose one feature - how many rounds need for learning