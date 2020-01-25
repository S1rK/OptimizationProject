from MachineLearning import learn
from SkyScanner import get_flights


if __name__ == "__main__":
    # get online flights
    flights = get_flights()
    # let the network to learn from them, and get the best flights out of them
    learn(flights)
