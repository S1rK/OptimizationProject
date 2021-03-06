import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch import optim
import random
from SkyScanner import attributes, flight_to_string
from GUI import get_priority
from typing import Callable, List, Tuple

# hyper : optimazer + Learning Rate
# test 1: 5 random flights vs. 5 best of each features
# test 2: auto agent - always choose one feature - how many rounds need for learning
# remember - need to be symetric, can put data1+data2=1, or data2+data1=0

features = len(attributes)
input = 2 * features
epoch = 3
learn_feature = 0


def rank_by_predicted_comparator(flights: np.array, predict: Callable[[np.array], int]):
    """

    :param flights:
    :param comparator:
    :return:
    """
    length = len(flights)
    flights_ranks = [0 for _ in range(length)]
    for i in range(0, length):
        f1 = flights[i]
        for j in range(0, length):
            if i != j:
                f2 = flights[j]
                x = np.concatenate([f1, f2], axis=0)
                if predict(x) == 0:
                    flights_ranks[i] += 1
                else:
                    flights_ranks[j] += 1
    return [f for f, _ in sorted(zip(flights, flights_ranks), key=lambda pair: pair[1], reverse=True)]


def sum_automatic_user(flight1: np.array, flight2: np.array) -> int:
    """
    An automatic user deciding only by min sum of attributes of flights.
    :param flight1: first flight.
    :param flight2: second flight.
    :return: 0 if it prefer flight1, 1 if it prefer flight2.
    """
    s1 = flight1[0] + flight1[1] + flight1[2]
    s2 = flight2[0] + flight2[1] + flight2[2]
    if s1 > s2:
        return 1
    return 0


def greedy_automatic_user(flight1: np.array, flight2: np.array) -> int:
    """
    An automatic user deciding only by min price of flights.
    :param flight1: first flight.
    :param flight2: second flight.
    :return: 0 if it prefer flight1, 1 if it prefer flight2.
    """
    if flight1[0] == flight2[0]:
        return int(flight1[1] > flight2[1])
    return int(flight1[0] > flight2[0])


def price_per_duration_automatic_user(flight1: np.array, flight2: np.array) -> int:
    """
    An automatic user deciding only by min (1 + number of connections) * price to duration ratio
    (min (1+number of connections)*price/duration) of flights.
    :param flight1: first flight.
    :param flight2: second flight.
    :return: 0 if it prefer flight1, 1 if it prefer flight2.
    """
    f1 = (1 + flight1[2]) * flight1[1] + flight1[0]
    f2 = (1 + flight2[2]) * flight2[1] + flight2[0]
    return int(f1 > f2)


def ranking_flights_by_automatic_user(flights: np.array, automatic_user: Callable[[np.array, np.array], int]):
    # length = len(flights)
    # for i in range(0, length):
    #     max_index = i
    #     for j in range(i, length):
    #         if automatic_user(flights[max_index], flights[j]) == 1:
    #             max_index = j
    #     # temp = flights[i]
    #     # flights[i] = flights[max_index]
    #     # flights[max_index] = temp
    #     # swap between them
    #     flights[i], flights[max_index] = flights[max_index], flights[i]
    # return flights

    from functools import cmp_to_key
    return sorted(flights, key=cmp_to_key(lambda x, y: 1 if automatic_user(x, y) is 1 else -1))


class net(nn.Module):
    def __init__(self, ):
        super(net, self).__init__()
        self.features = input
        self.fc2 = nn.Linear(input, int(input / 2))
        self.fc3 = nn.Linear(int(input / 2), 2)

    def forward(self, x):
        x = torch.FloatTensor(x)
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

    def from_rank_to_train_set(self, flights):
        data_to_learn = np.ndarray((0, features * 2))
        labels = np.ndarray((0, 1), dtype=int)
        for i in range(0, len(flights)):
            for j in range(i + 1, len(flights)):
                if i != j:
                    c1 = np.concatenate([flights[i], flights[j]], axis=0)
                    data_to_learn = np.append(data_to_learn, [c1], axis=0)
                    labels = np.append(labels, [[0]], axis=0)
                    c2 = np.concatenate([flights[j], flights[i]], axis=0)
                    data_to_learn = np.append(data_to_learn, [c2], axis=0)
                    labels = np.append(labels, [[1]], axis=0)
        data_to_learn = torch.FloatTensor(data_to_learn)
        labels = torch.IntTensor(labels)
        labels = labels.to(dtype=torch.int64)
        training_set = [(x, y) for x, y in zip(data_to_learn, labels)]
        return training_set


class SVM(object):
    def __init__(self, _eta=0.001, lambd=0.001):
        self.num_of_class = 2
        self._w = np.zeros((self.num_of_class, input))
        self._b = np.zeros(self.num_of_class)
        self.eta = _eta
        self._lambda = lambd

    def train_svm(self, training_set):
        x = len(training_set)
        for k in range(0, 5):
            np.random.shuffle(training_set)
            for x, y in training_set:
                y_hat = int(self.prediction(x))
                if y_hat != y:
                    self._w[y, :] = (1 - self.eta * self._lambda) * self._w[y, :] + self.eta * x
                    self._w[y_hat, :] = (1 - self.eta * self._lambda) * self._w[y_hat, :] - self.eta * x
                    a = self._w[y_hat, :]
                    for j in range(0, self.num_of_class):
                        if j != y and j != y_hat:
                            self._w[j, :] = (1 - self.eta * self._lambda) * self._w[j, :]

    def prediction(self, x):
        res = np.dot(self._w, np.transpose(x))
        return np.argmax(res)

    def ranking_flights(self, flights):
        return rank_by_predicted_comparator(flights, self.prediction)

    def from_rank_to_train_set(self, flights):
        data_to_learn = np.ndarray((0, features * 2))
        labels = np.ndarray((0, 1), dtype=int)
        for i in range(0, len(flights)):
            for j in range(i + 1, len(flights)):
                if i != j:
                    c1 = np.concatenate([flights[i], flights[j]], axis=0)
                    data_to_learn = np.append(data_to_learn, [c1], axis=0)
                    labels = np.append(labels, [[0]], axis=0)
                    c2 = np.concatenate([flights[j], flights[i]], axis=0)
                    data_to_learn = np.append(data_to_learn, [c2], axis=0)
                    labels = np.append(labels, [[1]], axis=0)
        training_set = [(x, y) for x, y in zip(data_to_learn, labels)]
        return training_set


def create_training_set_by_auto_user(flights: np.array,
                                     auto_user_comparator: Callable[[np.array, np.array], int]) -> List[
                                     Tuple[np.array, np.array]]:
    data_to_learn = np.ndarray((0, features * 2))
    labels = np.ndarray((0, 1), dtype=int)
    length = len(flights)
    for i in range(0, length):
        f1 = flights[i]
        for j in range(i + 1, length):
            f2 = flights[j]
            f1_to_f2 = auto_user_comparator(f1, f2)
            f2_to_f1 = auto_user_comparator(f2, f1)
            if f1_to_f2 != f2_to_f1:
                c1 = np.concatenate([f1, f2], axis=0)
                data_to_learn = np.append(data_to_learn, [c1], axis=0)
                labels = np.append(labels, [[f1_to_f2]], axis=0)
                c2 = np.concatenate([f2, f1], axis=0)
                data_to_learn = np.append(data_to_learn, [c2], axis=0)
                labels = np.append(labels, [[f2_to_f1]], axis=0)
    return list(zip(data_to_learn, labels))


def evaluate_ranking(real_ranking, predicted_ranking):
    ham_dis = 0
    for i in range(len(real_ranking)):
        ham_dis += abs(predicted_ranking.index(real_ranking[i]) - i)
    return ham_dis


def evaluate_model_by_auto_user(validation_set: np.array, auto_user_comparator: Callable[[np.array, np.array], int],
                                model_prediction: Callable[[np.array, np.array], int]) -> float:
    length = len(flights)
    num_of_iter = 0
    success = 0
    for i in range(0, length):
        f1 = flights[i]
        for j in range(0, length):
            if i != j:
                f2 = flights[j]

                x = np.concatenate([f1, f2], axis=0)
                if auto_user_comparator(f1, f2) == model_prediction(x):
                    success += 1

                x = np.concatenate([f2, f1], axis=0)
                if auto_user_comparator(f2, f1) == model_prediction(x):
                    success += 1

                num_of_iter += 2
    return success / num_of_iter


# def old_learn(flights: np.array, auto_user_comparator: Callable[[np.array, np.array], int] = greedy_automatic_user):
#     # shuffle the flights
#     np.random.shuffle(flights)
#
#     print(f"num flights: {len(flights)}")
#
#     _net = net()
#     optimizer = optim.SGD(_net.parameters(), lr=0.1)
#     _svm = SVM()
#
#     for i in range(0, epoch):
#         svm_rank = _svm.ranking_flights(flights)
#         net_rank = _net.ranking_flights(flights)
#
#         print("svm eval:")
#         print(evaluate_model_by_auto_user(flights, auto_user_comparator, _svm.prediction))
#         print("net eval:")
#         print(evaluate_model_by_auto_user(flights, auto_user_comparator, _net.prediction))
#
#         svm_top_5 = ranking_flights_by_automatic_user(svm_rank, auto_user_comparator)
#         net_top_5 = ranking_flights_by_automatic_user(net_rank, auto_user_comparator)
#
#         # TODO: put here method for evaluation the ranking
#         """
#         SHOULD WE REALLY EVALUATE ONLY BY RANKING, AND NOT BY PARIS AND THEIR COMPARING?
#         """
#
#         svm_set_train = _svm.from_rank_to_train_set(svm_top_5)
#         net_set_train = _net.from_rank_to_train_set(net_top_5)
#
#         _svm.train_svm(svm_set_train)
#         _net.train_net(optimizer, net_set_train)
#
#     print("---------svm recommend-----------")
#     print("\n".join([flight_to_string(f) for f in svm_rank[:5]]))
#     print("---------net recommend-----------")
#     print("\n".join([flight_to_string(f) for f in net_rank[:5]]))
#
#     print("here!")


def learn(flights: np.array, auto_user_comparator: Callable[[np.array, np.array], int] = greedy_automatic_user):
    print(f"num flights: {len(flights)}")

    _net = net()
    optimizer = optim.SGD(_net.parameters(), lr=0.1)
    _svm = SVM()

    for i in range(0, epoch):
        # svm_rank = _svm.ranking_flights(flights)
        # net_rank = _net.ranking_flights(flights)

        # print("svm eval:")
        # print(evaluate_model_by_auto_user(flights, auto_user_comparator, _svm.prediction))
        # print("net eval:")
        # print(evaluate_model_by_auto_user(flights, auto_user_comparator, _net.prediction))

        # svm_top_5 = ranking_flights_by_automatic_user(svm_rank, auto_user_comparator)
        # net_top_5 = ranking_flights_by_automatic_user(net_rank, auto_user_comparator)

        # train the models with the training set
        # svm_set_train = _svm.from_rank_to_train_set(training_set)
        # net_set_train = _net.from_rank_to_train_set(training_set)

        # TODO: put here method for evaluation the ranking
        """
        SHOULD WE REALLY EVALUATE ONLY BY RANKING, AND NOT BY PARIS AND THEIR COMPARING?
        """

        # shuffle the flights
        np.random.shuffle(flights)

        # get the current flights
        current_flights = flights[:len(flights)//2]

        # rank the current flights by the auto user comparator
        ranked_flights = ranking_flights_by_automatic_user(current_flights, auto_user_comparator)

        # create a training set by the ranked flights and the auto user comparator
        training_set = create_training_set_by_auto_user(ranked_flights, auto_user_comparator)

        # train the models with the training set
        _svm.train_svm(training_set)
        _net.train_net(optimizer, training_set)

        print("svm eval:")
        print(evaluate_model_by_auto_user(flights, auto_user_comparator, _svm.prediction))
        print("net eval:")
        print(evaluate_model_by_auto_user(flights, auto_user_comparator, _net.prediction))

    print("---------svm recommend-----------")
    svm_rank = rank_by_predicted_comparator(flights, _svm.prediction)
    print("\n".join([flight_to_string(f) for f in svm_rank[:5]]))
    print("---------net recommend-----------")
    net_rank = rank_by_predicted_comparator(flights, _net.prediction)
    print("\n".join([flight_to_string(f) for f in net_rank[:5]]))

    print("here!")


if __name__ == "__main__":
    from SkyScanner import get_flights

    flights = get_flights()
    np.random.shuffle(flights)
    flights = flights[:100]

    _svm = SVM()
    print("got flights")

    ranked_flights = ranking_flights_by_automatic_user(flights, greedy_automatic_user)
    print("ranked flights")

    training_set = create_training_set_by_auto_user(ranked_flights, greedy_automatic_user)
    print("created training set")

    # from random import shuffle
    # shuffle(training_set)
    # training_set = training_set[:5]

    _svm.train_svm(training_set)
    print("trained model")

    svm_rank = _svm.ranking_flights(flights)

    print("svm eval:")
    print(evaluate_model_by_auto_user(flights, greedy_automatic_user, _svm.prediction))

    print("---------svm recommend-----------")
    svm_rank = rank_by_predicted_comparator(ranked_flights, _svm.prediction)
    print("\n".join([flight_to_string(f) for f in svm_rank[:5]]))

    # for i in range(len(svm_rank)-1):
    #     f1 = svm_rank[i]
    #     for j in range(i+1, len(svm_rank)-1):
    #         f2 = svm_rank[j]
    #         # print(
    #         #     f"svm({i}, {j}) = {_svm.prediction(np.concatenate([f1, f2], axis=0))}\n")
    #         # print(
    #         #     f"svm({j}, {i}) = {_svm.prediction(np.concatenate([f2, f1], axis=0))}\n")
    #         if _svm.prediction(np.concatenate([f1, f2], axis=0)) != 1:
    #             print("---------NOT 1---------")
    #             print(flight_to_string(f1))
    #             print(flight_to_string(f2))
    #             print("-----------------------")

    print("---------result-----------")
    ranked = ranking_flights_by_automatic_user(ranked_flights, greedy_automatic_user)
    ranked = ranked[:5]
    print("\n".join([flight_to_string(f) for f in ranked[:5]]))
    print("--------------------------")

    # exit
    exit()

    learn(flights, greedy_automatic_user)

    print("---------result-----------")
    ranked = ranking_flights_by_automatic_user(flights, greedy_automatic_user)
    ranked = ranked[:5]
    print("\n".join([flight_to_string(f) for f in ranked[:5]]))
    print("--------------------------")
