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


class net(nn.Module):
    def __init__(self, ):
        super(net, self).__init__()
        self.features = input
        self.fc2 = nn.Linear(input, int(input/2))
        self.fc3 = nn.Linear(int(input/2), 2)

    def forward(self, x):
        x = x.view(-1, self.features)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def train_net(model, optimizer, training_set):
    model.train()
    optimizer.zero_grad()
    train_loss = 0
    for x, y in training_set:
        output = model(x)
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
        train_loss += F.nll_loss(output, y, size_average=False).item()

    # print(train_loss/len(training_set))

def test_net(model, training_set):
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
    # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, test_correct,
    #                                                                          n,
    #                                                                          100. * test_correct / n))
    return int(test_correct) / n

def prediction(model, data):
    model.eval()
    output = model(data)
    pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
    return pred

def check(idx, arr):
    for i in range(0,len(arr)):
        if arr[i]==idx:
            return False
    return True

def auto_agent_net(flights, _net, optimizer,learn_idx):
    n = len(flights)
    # pre-learning
    data_to_learn = np.ndarray((0, features * 2))
    labels = np.ndarray((0, 1), dtype=int)
    for i in range(0, features):
        for j in range(0, features):
            if i != j:
                c = np.concatenate([flights[learn_idx[learn_feature]], flights[learn_idx[i]]], axis=0)
                data_to_learn = np.append(data_to_learn, [c], axis=0)
                labels = np.append(labels, [[1]], axis=0)
                c = np.concatenate([flights[learn_idx[i]], flights[learn_idx[learn_feature]]], axis=0)
                data_to_learn = np.append(data_to_learn, [c], axis=0)
                labels = np.append(labels, [[0]], axis=0)
    # learning
    data_to_learn = torch.FloatTensor(data_to_learn)
    labels = torch.IntTensor(labels)
    labels = labels.to(dtype=torch.int64)
    training_set = [(x, y) for x, y in zip(data_to_learn, labels)]
    # print(training_set)
    for i in range(0, epoches):
        # print("epoch: " + str(i))
        train_net(_net, optimizer, training_set)

    # pre - testing
    data_to_test = np.ndarray((0, features * 2))
    labels_test = np.ndarray((0, 1), dtype=int)
    for i in range(0, n):
        for j in range(0, n):
            if i != j and check(i, learn_idx):
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
    return test_net(_net, test_set)


def auto_agent_svm(flights,learn_idx,lr):
    print("svm...")


def get_smart_priorty(flights,num):
    ind_best_each = np.zeros(num, dtype=int)
    n = len(flights)

    run = min(features,num)
    for i in range(0, n-(num-features-1)):
        for j in range(0, run):
            s = ind_best_each[j]
            if flights[s][j] > flights[i][j]:
                ind_best_each[j] = i
    if features<num:
        s = max(ind_best_each)
        for j in range(features, num):
            ind_best_each[j] = s+j-features+1;

    return ind_best_each

def learn(flights):
    num_of_flights = len(flights)
    print("num flights: ")
    print(num_of_flights)

    # TODO: put this priority vector to use
    #priority = np.array([flights[i] for i in get_priority([flight_to_string(flight) for flight in flights[:5]])])

    rand_priority = np.random.randint(0, num_of_flights-1, 5)
    smart_priority = get_smart_priorty(flights,5)



    runs = 1
    lr_p = np.array([0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9])
    best_lr1 = np.zeros(len(lr_p))
    best_lr2 = np.zeros(len(lr_p))
    # test 1:nets: smart choise vs. rand choise

    for lr in range(0,1):
        s1 = 0
        s2 = 0
        for i in range(0, runs):
            _net1 = net()
            optimizer1 = optim.SGD(_net1.parameters(), lr=lr_p[lr])
            _net2 = net()
            optimizer2 = optim.SGD(_net2.parameters(), lr=lr_p[lr])
            s1+=auto_agent_net(flights, _net1, optimizer1,rand_priority)
            s2+=auto_agent_net(flights, _net2, optimizer2,smart_priority)
        best_lr1[lr] = s1/runs
        best_lr2[lr] = s2/runs
    print(best_lr1)
    print(best_lr2)
    print(max(best_lr1))
    print(max(best_lr2))


    # test 2:svm: smart choise vs. rand choise
    auto_agent_svm(flights,rand_priority,0.8)
    auto_agent_svm(flights, smart_priority, 0.8)




    print("here!")
