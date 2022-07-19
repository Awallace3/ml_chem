from models import bpnn_test
from setup import collect_data
import pprint
import os
from models.bpnn_test import write_pickle, read_pickle
from models.structs import paths, acsf_Gs, acsf_model, results, nn_props


pp = pprint.PrettyPrinter(indent=4)

def bpnn(acsf_model: acsf_model, train=True):
    if not os.path.exists(acsf_model.paths.data_path):
        xs, ys = collect_data(
            acsf_model.num_molecules,
            progress=True,
            Rc=acsf_model.acsf_Gs.Rc,
            G2_params=acsf_model.acsf_Gs.G2_params,
            G4_params=acsf_model.acsf_Gs.G4_params,
        )
        data = [xs, ys]
        write_pickle(data, acsf_model.paths.data_path)
    else:
        print("found data at %s" % acsf_model.paths.data_path)
        data = read_pickle(acsf_model.paths.data_path)
        xs, ys = data[0], data[1]

    if train:
        bpnn_test.atomic_nn(
            xs,
            ys,
            acsf_model
        )
    else:
        bpnn_test.test_atomic_nn(
            xs,
            ys,
            acsf_model
        )


def train_bpnn():
    data_path = "./data/t11_dataset.pickle"
    G2_params = [(0.4, 0.2), (0.6, 0.2)]
    G4_params = [(0.4, 2, 1), (0.6, 2, 1), (0.6, 2, -1)]
    Rc = 5.0

    if not os.path.exists(data_path):
        G2_params = [(0.4, 0.2), (0.6, 0.2)]
        G4_params = [(0.4, 2, 1), (0.6, 2, 1), (0.6, 2, -1)]
        Rc = 5.0
        print("collecting data...")
        xs, ys = collect_data(100,
                              progress=True,
                              Rc=Rc,
                              G2_params=G2_params,
                              G4_params=G4_params)
        data = [xs, ys]
        write_pickle(data, data_path)
    else:
        data = read_pickle(data_path)
        xs, ys = data[0], data[1]
    print('data collected')
    epochs = 300
    lr = 1e-3
    bpnn_test.atomic_nn(xs, ys, epochs=epochs, learning_rate=lr)
    return


def test_bpnn(model_name="t19", data_path="data/t0.pickle"):
    data_path = "data/t0.pickle"
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
    bpnn_test.test_atomic_nn(xs, ys, model_name=model_name, scales=False)
    return


def remove_model(acsf_model: acsf_model):
    pickle_path = "data/models.pickle"
    data = read_pickle(pickle_path)
    pos = -1
    print(data)
    for n, i in enumerate(data):
        if acsf_model.name == i.name:
            print(acsf_model.paths.data_path)
            # os.remove(acsf_model.paths.data_path)
            # os.remove(acsf_model.paths.plot_path)
            # os.remove(acsf_model.paths.model_path)
            print('removing')
            pos = n
    if pos != -1:
        data.pop(n)
        print(data, n)
        write_pickle(data, pickle_path)


def read_model(acsf_model: acsf_model):
    pickle_path = "data/models.pickle"
    if os.path.exists(pickle_path):
        data = read_pickle(pickle_path)
        for i in data:
            if acsf_model.name == i.name:
                print("found %s with %d molecules" % (i.name, i.num_molecules))
                print("Gs params...")
                print(acsf_model.acsf_Gs)
                return i
        data.append(acsf_model)
        write_pickle(data, pickle_path)
        return acsf_model
    else:
        data = [acsf_model]
        write_pickle(data, pickle_path)
        return acsf_model


def main():
    # model_name = "t2"
    model_name = "t6"
    G2_params = [
        (0.4, 0.2),
        (0.6, 0.2),
    ]
    G4_params = [
        (0.4, 2, 1),
        (0.6, 2, 1),
        (0.6, 2, -1),
    ]
    Rc = 5.0
    Gs = acsf_Gs(
        G2_params,
        G4_params,
        Rc,
    )
    nn_p = nn_props(
        nodes=[64, 64, 32, 1],
        epochs=100,
        learning_rate=0.001,
        batch_size=32,
    )
    # t0 = 1e4
    # t3 = 1e6
    # t6 = 1e2
    p = paths(
        data_path="data/t3.pickle",
        model_path="results/%s" % model_name,
        linear_model="results/%s_linear" % model_name,
        plot_path="plots/%s" % model_name,
    )
    m = acsf_model(
        model_name,
        num_molecules=100000,
        acsf_Gs=Gs,
        paths=p,
        nn_props=nn_p,
        results=results(),
    )
    # remove_model(m)
    # m = read_model(m)
    pp.pprint(m)

    bpnn(
        m,
        train=True,
        # train=False,
    )

    # train_bpnn()
    # test_bpnn(model_name, acsf_model.paths.data_path)

    return


"""
ideas...
1. scaling
    - perhaps need to run 1 epoch to figure out average of what each element
      should scale to?
    - take crude approximation as each element contributing equally to total
      energy to rescale
        - refine this guess as well?
"""

if __name__ == "__main__":
    main()
