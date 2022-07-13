from setup import collect_data
import pprint
from models import bpnn_test
from models.bpnn_test import write_pickle, read_pickle
import os

pp = pprint.PrettyPrinter(indent=4)


def train_bpnn():
    G2_params = [(0.4, 0.2), (0.6, 0.8), (0.6, 0.2), (0.8, 0.5)]
    G2_params = [(0.4, 0.2), (0.6, 0.2)]
    G4_params = [(0.4, 2, 1), (0.6, 2, 1), (0.6, 2, -1)]
    Rc = 5.0
    print("collecting data...")
    xs, ys = collect_data(4,
                          progress=True,
                          Rc=Rc,
                          G2_params=G2_params,
                          G4_params=G4_params)
    print('data collected')
    epochs = 500
    lr = 1e-3
    bpnn_test.atomic_nn(xs, ys, epochs=epochs, learning_rate=lr)


def test_bpnn():

    data_path = "./data/t2_dataset.pickle"
    G2_params = [(0.4, 0.2), (0.6, 0.8), (0.6, 0.2), (0.8, 0.5)]
    if not os.path.exists(data_path):

        G2_params = [(0.4, 0.2), (0.6, 0.2)]
        G4_params = [(0.4, 2, 1), (0.6, 2, 1), (0.6, 2, -1)]
        Rc = 5.0
        print("collecting data...")
        xs, ys = collect_data(10000,
                              progress=True,
                              Rc=Rc,
                              G2_params=G2_params,
                              G4_params=G4_params)
        data = [xs, ys]
        write_pickle(data, data_path)
    else:
        data = read_pickle(data_path)
        xs, ys = data[0], data[1]
    bpnn_test.test_atomic_nn(xs, ys)
    return


def main():
    train_bpnn()
    # test_bpnn()

    return


if __name__ == "__main__":
    main()
