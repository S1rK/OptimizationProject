import MachineLearning
import GUI
import SkyScanner


if __name__ == "__main__":
    # get online flights
    flights = SkyScanner.get_flights()
    # let the network to learn from them, and get the best flights out of them
    MachineLearning.learn(flights)
