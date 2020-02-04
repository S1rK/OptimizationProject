import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch import optim
import random
from SkyScanner import attributes, flight_to_string
from GUI import get_priority
from typing import Callable

# hyper : optimazer + Learning Rate
# test 1: 5 random flights vs. 5 best of each features
# test 2: auto agent - always choose one feature - how many rounds need for learning
# remember - need to be symetric, can put data1+data2=1, or data2+data1=0

features = len(attributes)
input = 2 * features
epoch = 10
learn_feature = 0


def rank_by_predicted_comparator(flights, comparator: Callable):
    length = len(flights)
    flights_ranks = [0 for _ in range(length)]
    for i in range(0, length):
        for j in range(i, length):
            if comparator(flights[i], flights[j]) == 0:
                flights_ranks[i] += 1
            else:
                flights_ranks[j] += 1
    return [f for f, _ in sorted(zip(flights, flights_ranks), key=lambda pair: pair[1])]


def sum_automatic_user(flight1, flight2) -> int:
    s1 = flight1[0] + flight1[1] + flight1[2]
    s2 = flight2[0] + flight2[1] + flight2[2]
    if s1 > s2:
        return 1
    return 0


def greedy_automatic_user(flight1, flight2) -> int:
    return int(flight1[0]>flight2[0])


def ranking_flights_by_automatic_user(flights, automatic_user: Callable):
    length = len(flights)
    for i in range(0, length):
        max_index = i
        for j in range(i, length):
            if automatic_user(flights[max_index], flights[j]) == 1:
                max_index = j
        temp = flights[i]
        flights[i] = flights[max_index]
        flights[max_index] = temp
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

    def ranking_flights(self, flights):
        return rank_by_predicted_comparator(flights, self.prediction)


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

    def ranking_flights(self, flights):
        return rank_by_predicted_comparator(flights, self.prediction)


def from_rank_to_train_set(flights):
    data_to_learn = np.ndarray((0, features * 2))
    labels = np.ndarray((0, 1), dtype=int)
    for i in range(0, len(flights)):
        for j in range(i + 1, len(flights)):
            if i != j:
                c1 = np.concatenate([flights[i], flights[j]], axis=0)
                data_to_learn = np.append(data_to_learn, [c1], axis=0)
                labels = np.append(labels, [[1]], axis=0)
                c2 = np.concatenate([flights[j], flights[i]], axis=0)
                data_to_learn = np.append(data_to_learn, [c2], axis=0)
                labels = np.append(labels, [[0]], axis=0)
    data_to_learn = torch.FloatTensor(data_to_learn)
    labels = torch.IntTensor(labels)
    labels = labels.to(dtype=torch.int64)
    training_set = [(x, y) for x, y in zip(data_to_learn, labels)]
    return training_set


def evaluate_ranking(real_ranking, predicted_ranking):
    ham_dis = 0
    for i in range(len(real_ranking)):
        ham_dis += abs(predicted_ranking.index(real_ranking[i]) - i)
    return ham_dis


def learn(flights):
    num_of_flights = len(flights)

    good_ranking_flights = ranking_flights_by_automatic_user(flights, sum_automatic_user)

    _net = net()
    optimizer = optim.SGD(_net.parameters(), lr=0.01)

    _svm = SVM()

    for i in range(0, epoch):
        svm_rank = _svm.ranking_flights(flights)
        net_rank = _net.ranking_flights(flights)

        svm_top_5 = ranking_flights_by_automatic_user(svm_rank[0 - 5], sum_automatic_user)
        net_top_5 = ranking_flights_by_automatic_user(net_rank[0 - 5], sum_automatic_user)

        # TODO: put here method for evaluation the ranking
        """
        SHOULD WE REALLY EVALUATE ONLY BY RANKING, AND NOT BY PARIS AND THEIR COMPARING?
        """
        svm_eval = evaluate_ranking(good_ranking_flights, svm_top_5)
        net_eval = evaluate_ranking(good_ranking_flights, net_top_5)

        svm_set_train = from_rank_to_train_set(svm_top_5)
        net_set_train = from_rank_to_train_set(net_top_5)

        _svm.train_svm(svm_set_train)
        _net.train_net(optimizer, net_set_train)

    print("here!")


if __name__ == "__main__":
    from SkyScanner import get_flights
    flights = get_flights()
    print(f"got {len(flights)} flights")
    np.random.shuffle(flights)
    fs = flights[:5]
    fs2 = fs
    ranked = ranking_flights_by_automatic_user(fs, sum_automatic_user)
    print("\n".join([flight_to_string(f) for f in ranked]))
    print("--------------------------")
    ranked = ranking_flights_by_automatic_user(fs2, greedy_automatic_user)
    print("\n".join([flight_to_string(f) for f in ranked]))

