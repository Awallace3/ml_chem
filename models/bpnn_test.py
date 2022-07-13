import torch
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob
import pickle
import matplotlib.pyplot as plt


def rescale_targets(ys: []):
    """
    Rescales the target values to be smaller values
    """
    np_y = np.array(ys)
    mu = np.mean(np_y)
    std = np.std(np_y)
    return [(i - mu) / std for i in ys], mu, std


def write_pickle(data, fname='data.pickle'):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(fname='data.pickle'):
    with open(fname, 'rb') as handle:
        return pickle.load(handle)


def data_to_tensors(xs, ys):
    """
    Used to restructure data coming from setup for pytorch usage
    """
    for n, i in enumerate(xs):
        i = np.float32(i)
        xs[n] = torch.from_numpy(i)
        ys[n] = torch.tensor(ys[n], dtype=torch.float32)
    return xs, ys


def prepare_data(xs, ys, train_size=0.3):
    """
    Splits dataset into train/test with restructuring datatypes to tensors
    """
    train_x, test_x, train_y, test_y = train_test_split(xs,
                                                        ys,
                                                        train_size=train_size,
                                                        test_size=1 -
                                                        train_size,
                                                        random_state=124)
    train_x, train_y = data_to_tensors(train_x, train_y)
    test_x, test_y = data_to_tensors(test_x, test_y)
    return train_x, test_x, train_y, test_y


class elementNet(torch.nn.Module):
    """
    Contains the default element schematic
    """

    def __init__(self, Gs_num: int):
        super().__init__()
        self.el = nn.Sequential(
            nn.Linear(Gs_num, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )


class atomicNet(torch.nn.Module):
    """
    Constructs NN for H, C, N, O, and F individually
    """

    def __init__(self, Gs_num: int):
        super().__init__()
        # self.h = elementNet(Gs_num)
        # self.c = elementNet(Gs_num)
        # self.n = elementNet(Gs_num)
        # self.o = elementNet(Gs_num)
        # self.f = elementNet(Gs_num)
        self.h = nn.Sequential(
            nn.Linear(Gs_num, 2 * Gs_num),
            nn.Linear(2 * Gs_num, Gs_num),
            nn.Linear(Gs_num, 1),
        )
        self.c = nn.Sequential(
            nn.Linear(Gs_num, 2 * Gs_num),
            nn.Linear(2 * Gs_num, Gs_num),
            nn.Linear(Gs_num, 1),
        )
        self.n = nn.Sequential(
            nn.Linear(Gs_num, 2 * Gs_num),
            nn.Linear(2 * Gs_num, Gs_num),
            nn.Linear(Gs_num, 1),
        )
        self.o = nn.Sequential(
            nn.Linear(Gs_num, 2 * Gs_num),
            nn.Linear(2 * Gs_num, Gs_num),
            nn.Linear(Gs_num, 1),
        )
        self.f = nn.Sequential(
            nn.Linear(Gs_num, 2 * Gs_num),
            nn.Linear(2 * Gs_num, Gs_num),
            nn.Linear(Gs_num, 1),
        )
        self.el = {1: self.h, 6: self.c, 7: self.n, 8: self.o, 9: self.f}

    def forward(self, x):
        k = int(x[0])
        v = self.el[k]
        v = v(x[1:])
        return v


def saveModel(model, path="./results"):
    """
    Saves current model without overwriting previous models
    """
    ms = glob(path + "/t*")
    save_path = path + "/t1"
    if len(ms) > 0:
        cnt = [int(i.split("/t")[-1]) for i in ms]
        v = max(cnt) + 1
        save_path = path + "/t%d" % v
        print('model saved to %s' % save_path)
    torch.save(model.state_dict(), save_path)
    return


def stats_results(differences: [], percentages: []):
    """
    Returns mean differences, mean percentages, and mean absolute error.
    """
    avg_dif = sum(differences) / len(differences)
    avg_perc = sum(percentages) / len(percentages)
    abs_dif = []
    for i in differences:
        if i < 0:
            abs_dif.append(-i)
        else:
            abs_dif.append(i)
    mae = sum(abs_dif) / len(abs_dif)
    print()
    print('Avg. Differences \t=\t%.4f Hartrees' % (avg_dif))
    print('Mean Abs. Error  \t=\t%.4f Hartrees' % (mae))
    print('Avg. Percentage E\t=\t%.4f Hartrees' % (avg_perc))


def test_atomic_nn(xs: [],
                   ys: [],
                   model_save_dir="./results",
                   model_name="t2",
                   Gs_num=30):
    """
    Tests a model without training. Use the same ACSF parameters as the model
    being used.
    """
    xs, test_xs, ys, test_ys = prepare_data(xs, ys, 0.8)
    model = atomicNet(Gs_num)
    model.load_state_dict(torch.load("%s/%s" % (model_save_dir, model_name)))
    model.eval()
    differences = []
    percentages = []
    E_model, E_target = [], []
    for n, x in enumerate(test_xs):
        y = test_ys[n]
        E_tot = 0
        for atom in x:
            E_i = model(atom)
            E_tot += E_i
        E_tot = E_tot[0]
        E_model.append(E_tot)
        E_target.append(y)

        dif = y - E_tot
        differences.append(dif)
        perc = abs(dif) / y
        percentages.append(perc)

    E_target = [float(i) for i in E_target]
    E_model = [float(i) for i in E_model]
    x = [i - 3200 for i in range(3500)]

    fig = plt.figure(dpi=400)
    plt.plot(E_target, E_model, 'r.', linewidth=0.1)
    plt.plot(x, x, 'k')
    plt.xlabel('Target Energy')
    plt.ylabel('Predicted Energy')
    plt.xlim([-3000, 0])
    plt.ylim([-3000, 0])
    plt.savefig("bpnn_results.png")

    stats_results(differences, percentages)
    return


def atomic_nn(xs: [],
              ys: [],
              model_save_dir="./results",
              epochs=300,
              learning_rate=1e-1):
    """
    Constructs a new model from a dataset of the structure...
    xs = [np.array[[atomic number, Gs...]]]
    ys = []
    """
    xs, test_xs, ys, test_ys = prepare_data(xs, ys, 0.8)

    Gs_num = xs[0].size()[1] - 1

    model = atomicNet(Gs_num)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("starting training...")
    ten_p = epochs // 10
    for epoch in range(epochs):
        for n, x in enumerate(xs):
            y = ys[n]
            E_tot = 0
            for atom in x:
                E_i = model(atom)
                E_tot += E_i
            E_tot = E_tot[0]
            loss = criterion(E_tot, y)
            if (epoch + 1) % ten_p == 0:
                print(epoch + 1, loss.item())
                # print(E_tot.size(), y.size())
                print("\t", float(E_tot), float(y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    saveModel(model, path=model_save_dir)

    with torch.no_grad():
        model.eval()
        differences = []
        percentages = []
        for n, x in enumerate(test_xs):
            y = test_ys[n]
            E_tot = 0
            for atom in x:
                E_i = model(atom)
                E_tot += E_i
            E_tot = E_tot[0]
            dif = y - E_tot
            differences.append(dif)
            perc = abs(dif) / y
            percentages.append(perc)

    stats_results(differences, percentages)
    return
