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
learn_feature = 1


class net(nn.Module):
    def __init__(self, ):
        super(net, self).__init__()
        self.features = input
        # self.fc2 = nn.Linear(input, int(input/2))
        self.fc3 = nn.Linear(input, 2)

    def forward(self, x):
        x = x.view(-1, self.features)
        # x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def train(model, optimizer, training_set):
    model.train()
    optimizer.zero_grad()
    train_loss = 0
    for x, y in training_set:
        output = model(x)
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
        train_loss += F.nll_loss(output, y, size_average=False).item()

    print(train_loss/len(training_set))


def test(model, training_set):
    model.eval()
    test_loss = 0
    test_correct = 0
    n = len(training_set)
    for x, y in training_set:
        output = model(x)
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        test_loss += F.nll_loss(output, y, size_average=False).item()
        test_correct += pred.eq(y.view_as(pred)).cpu().sum()
    test_loss /= n
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, test_correct,
                                                                             n,
                                                                             100. * test_correct / n))


def prediction(model, data):
    model.eval()
    output = model(data)
    pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
    return pred


def auto_agent_best_five(flights, _net, optimizer):
    # pre-learning
    ind_best_each = np.zeros(features, dtype=int)
    n = len(flights)
    for i in range(0, n):
        for j in range(0, features):
            s = ind_best_each[j]
            if flights[s][j] > flights[i][j]:
                ind_best_each[j] = i

    data_to_learn = np.ndarray((0, features * 2))
    labels = np.ndarray((0, 1), dtype=int)
    for i in range(0, features):
        if i != learn_feature:
            c = np.concatenate([flights[ind_best_each[learn_feature]], flights[ind_best_each[i]]], axis=0)
            data_to_learn = np.append(data_to_learn, [c], axis=0)
            labels = np.append(labels, [[1]], axis=0)
            c = np.concatenate([flights[ind_best_each[i]], flights[ind_best_each[learn_feature]]], axis=0)
            data_to_learn = np.append(data_to_learn, [c], axis=0)
            labels = np.append(labels, [[0]], axis=0)
    # learning
    data_to_learn = torch.FloatTensor(data_to_learn)
    labels = torch.IntTensor(labels)
    labels = labels.to(dtype=torch.int64)
    training_set = [(x, y) for x, y in zip(data_to_learn, labels)]
    # print(training_set)
    for i in range(0, epoches):
        print("epoch: " + str(i))
        train(_net, optimizer, training_set)

    # pre - testing
    data_to_test = np.ndarray((0, features * 2))
    labels_test = np.ndarray((0, 1), dtype=int)
    for i in range(0, n):
        for j in range(0, n):
            if i != j and i != ind_best_each[0] and i != ind_best_each[1] and i != ind_best_each[2] and j != \
                    ind_best_each[0] and j != ind_best_each[1] and j != ind_best_each[2]:
                c = np.concatenate([flights[i], flights[j]], axis=0)
                data_to_test = np.append(data_to_test, [c], axis=0)
                if flights[i][learn_feature] < flights[j][learn_feature]:
                    labels_test = np.append(labels_test, [[1]], axis=0)
                else:
                    labels_test = np.append(labels_test, [[0]], axis=0)
    # testing
    data_to_test = torch.FloatTensor(data_to_test)
    labels_test = torch.IntTensor(labels_test)
    labels_test = labels_test.to(dtype=torch.int64)
    test_set = [(x, y) for x, y in zip(data_to_test, labels_test)]
    test(_net, test_set)


def auto_agent_random_five(flights, _net, optimizer):
    # pre-learning
    cut = 7
    n = len(flights)

    data_to_learn = np.ndarray((0, features * 2))
    labels = np.ndarray((0, 1), dtype=int)
    for i in range(0, cut):
        for j in range(0, cut):
            if i != j:
                c = np.concatenate([flights[i], flights[j]], axis=0)
                data_to_learn = np.append(data_to_learn, [c], axis=0)
                if flights[i][learn_feature] < flights[j][learn_feature]:
                    labels = np.append(labels, [[1]], axis=0)
                else:
                    labels = np.append(labels, [[0]], axis=0)
    # learning
    data_to_learn = torch.FloatTensor(data_to_learn)
    labels = torch.IntTensor(labels)
    labels = labels.to(dtype=torch.int64)
    training_set = [(x, y) for x, y in zip(data_to_learn, labels)]
    # print(training_set)
    for i in range(0, epoches):
        print("epoch: " + str(i))
        train(_net, optimizer, training_set)

    # pre - testing
    data_to_test = np.ndarray((0, features * 2))
    labels_test = np.ndarray((0, 1), dtype=int)
    for i in range(cut, n):
        for j in range(cut, n):
            if i != j:
                c = np.concatenate([flights[i], flights[j]], axis=0)
                data_to_test = np.append(data_to_test, [c], axis=0)
                if flights[i][learn_feature] < flights[j][learn_feature]:
                    labels_test = np.append(labels_test, [[1]], axis=0)
                else:
                    labels_test = np.append(labels_test, [[0]], axis=0)
    # testing
    data_to_test = torch.FloatTensor(data_to_test)
    labels_test = torch.IntTensor(labels_test)
    labels_test = labels_test.to(dtype=torch.int64)
    test_set = [(x, y) for x, y in zip(data_to_test, labels_test)]
    test(_net, test_set)


def learn(flights):
    _net1 = net()
    optimizer1 = optim.SGD(_net1.parameters(), lr=0.8)
    _net2 = net()
    optimizer2 = optim.SGD(_net2.parameters(), lr=0.5)
    # num_of_flights = 100
    num_of_flights = len(flights)

    # TODO: put this priority vector to use
    priority = np.array([flights[i] for i in get_priority([flight_to_string(flight) for flight in flights[:5]])])
    exit()

    flights = np.random.rand(num_of_flights, features)
    np.random.shuffle(flights)

    auto_agent_best_five(flights, _net1, optimizer1)
    auto_agent_random_five(flights, _net2, optimizer2)

    print("here!")
