import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch import optim
import random
from SkyScanner import attributes, flight_to_string
from GUI import get_priority

# hyper : optimazer + Learning Rate
# test 1: 5 random flights vs. 5 best of each features
# test 2: auto agent - always choose one feature - how many rounds need for learning
# remember - need to be symetric, can put data1+data2=1, or data2+data1=0

features = len(attributes)
input = 2 * features
epoches = 10
learn_feature = 0


def fake_user_comparator(flight1, flight2):
    s1 = flight1[0] + flight1[1] + flight1[2]
    s2 = flight2[0] + flight2[1] + flight2[2]
    if s1 > s2:
        return 1
    return 0


def ranking_flights_by_user(flights):
    len = len(flights)
    for i in range(0, len):
        max = i
        for j in range(i, len):
            if fake_user_comparator(flights[max], flights[j]) == 1:
                max = j
        temp = flights[i]
        flights[i] = flights[max]
        flights[max] = temp
    return flights


class net(nn.Module):
    def __init__(self, ):
        super(net, self).__init__()
        self.features = input
        self.fc2 = nn.Linear(input, int(input / 2))
        self.fc3 = nn.Linear(int(input / 2), 2)

    def forward(self, x):
        x = x.view(-1, self.features)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def train_net(self, optimizer, training_set):
        self.train()
        optimizer.zero_grad()
        for x, y in training_set:
            output = self(x)
            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()

    def prediction(self, x):
        self.eval()
        output = self(x)
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        return pred

    def ranking_flights(self,flights):


class SVM(object):
    def __init__(self, eta=0.01, lambd=0.075):
        self.num_of_class = 2
        self.__w = np.zeros((self.num_of_class, input))
        self.__b = np.zeros(self.num_of_class)
        for i in range(self.num_of_class):
            self.__b[i] = 1.0
        self.__eta = eta
        self.__lambda = lambd

    def train_svm(self, training_set):
        for x, y in training_set:
            y_hat = int(self.prediction(x))
            self.__w[y, :] = (1 - self.__eta * self.__lambda) * self.__w[y, :] + self.__eta * x
            self.__w[y_hat, :] = (1 - self.__eta * self.__lambda) * self.__w[y, :] - self.__eta * x
            self.__b[y] = self.__b[y] + self.__eta
            self.__b[y_hat] = self.__b[y_hat] - self.__eta
            for i in range(self.num_of_class):
                if i != y and i != y_hat:
                    self.__w[i, :] = (1 - self.__eta * self.__lambda) * self.__w[i, :]

    def prediction(self, x):
        res = np.dot(self.__w, np.transpose(x))
        for i in range(self.num_of_class):
            res[i] = res[i] + self.__b[i]
        return np.argmax(res)


    def ranking_flights(self,flights):


def learn(flights):
    num_of_flights = len(flights)
    print("num flights: ")
    print(num_of_flights)

    _net = net()
    optimizer = optim.SGD(_net.parameters(), lr=0.01)

    print("here!")
