import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from .structs import acsf_model
import json
import os
import dataclasses


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


def prepare_minimization(xs, ys):
    xs, ys = data_to_tensors(xs, ys)
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


def elementNet(Gs_num):
    """
    Contains the default element schematic
    """
    return nn.Sequential(
        nn.Linear(Gs_num, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


class atomicNet(torch.nn.Module):
    """
    Constructs NN for H, C, N, O, and F individually
    """

    def __init__(self, Gs_num: int):
        super().__init__()
        self.h = elementNet(Gs_num)
        self.c = elementNet(Gs_num)
        self.n = elementNet(Gs_num)
        self.o = elementNet(Gs_num)
        self.f = elementNet(Gs_num)
        self.el = {1: self.h, 6: self.c, 7: self.n, 8: self.o, 9: self.f}

    def forward(self, x):
        k = int(x[0])
        v = self.el[k]
        v = v(x[1:])
        return v


def saveModel_linear(
    model,
    acsf_model: acsf_model,
):
    """saves linear piece for regression"""

    save_path = acsf_model.paths.linear_model
    torch.save(model.state_dict(), save_path)
    return


def saveModel_ACSF_model(
    model,
    acsf_model: acsf_model,
    scales=[1, 1],
):
    """
    Saves current model
    """

    save_path = acsf_model.paths.model_path
    torch.save(model.state_dict(), save_path)
    # with open(save_path + "_scale", 'w') as fp:
    #     for i in scales:
    #         fp.write(str(i))
    #         fp.write('\n')
    return


def stats_results(differences: [], percentages: [], acsf_obj: acsf_model):
    """
    Returns mean differences, mean percentages, and mean absolute error.
    """
    max_d = max(differences)
    avg_dif = sum(differences) / len(differences)
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
    print('Max Error        \t=\t%.4f Hartrees' % (max_d))
    acsf_model.results.avg_dif = avg_dif
    acsf_model.results.mae = mae
    acsf_model.results.max_d = max_d
    json_p = acsf_obj.paths.model_path + ".json"
    print(json_p)
    with open(json_p, 'w') as f:
        f.write(json.dumps(dataclasses.asdict(acsf_obj), indent=4))
    return


def read_scales(path):
    with open(path, 'r') as f:
        data = f.readlines()
    data = [float(i.replace("\n", "")) for i in data]
    return data


def test_atomic_nn(
    xs: [],
    ys: [],
    acsf_obj: acsf_model,
    scales=True,
):
    """
    Tests a model without training. Use the same ACSF parameters as the model
    being used.
    """
    # rescales to have mean=0 and std=1
    # if scales:
    #     scale_path = "%s_scale" % (acsf_obj.paths.model_path)
    #     d = read_scales(scale_path)
    #     mu, std = d[0], d[1]
    #     scales = [mu, std]

    xs, test_xs, ys, test_ys = prepare_data(xs, ys, 0.8)
    Gs_num = xs[0].size()[1] - 1
    print(xs, ys)

    model = atomicNet(Gs_num)
    model.load_state_dict(torch.load(acsf_obj.paths.model_path))

    with torch.no_grad():
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
            # if scales:
            #     E_tot = E_tot * std + mu
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
    # plt.xlim([-3000, 0])
    # plt.ylim([-3000, 0])
    plt.savefig(acsf_obj.paths.plot_path)

    stats_results(differences, percentages, acsf_obj)
    return


# for xs, ragged inputs
# - j

# 1. compute error each epoch, average training errors from
# epoch
# 2. eval validation set each epoch
# 3. savemodel when validation error is lowest
# 4.


def elementNetMinimizeInput(Gs_num):
    """
    Contains the default element schematic
    """
    return nn.Sequential(nn.Linear(Gs_num, 1), )


class atomicNetMinimizeInput(torch.nn.Module):
    """
    Constructs NN for H, C, N, O, and F individually
    """

    def __init__(self, Gs_num: int):
        super().__init__()
        self.h = elementNetMinimizeInput(Gs_num)
        self.c = elementNetMinimizeInput(Gs_num)
        self.n = elementNetMinimizeInput(Gs_num)
        self.o = elementNetMinimizeInput(Gs_num)
        self.f = elementNetMinimizeInput(Gs_num)
        self.el = {1: self.h, 6: self.c, 7: self.n, 8: self.o, 9: self.f}

    def forward(self, x):
        k = int(x[0])
        v = self.el[k]
        v = v(x[1:])
        return v


class linearRegression(torch.nn.Module):

    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


def minimize_error(xs: [], ys: [], acsf_obj: acsf_model):
    """
    Minimizes error before neural network training.
    """
    # print(type(xs), type(xs[0]), np.shape(xs[0]))
    # print(type(ys), type(ys[0]))
    print('Number of Molecules:', len(xs))
    xs, ys = prepare_minimization(xs, ys)
    Gs_num = xs[0].size()[1] - 1
    if not os.path.exists(acsf_obj.paths.linear_model):
        model = atomicNet(Gs_num)
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        batch_size = 1
        batch = 0
        E_totals = torch.zeros(batch_size)
        local_ys = torch.zeros(batch_size)
        for epoch in range(1):
            print("linear")
            for n, x in enumerate(xs):
                y = ys[n]
                E_tot = 0
                for atom in x:
                    E_i = model(atom)
                    E_tot += E_i
                E_tot = E_tot[0]
                E_totals[batch] = E_tot
                local_ys[batch] = y
                batch += 1
                if batch == batch_size:
                    batch = 0
                    # loss = criterion(E_tot, y)
                    loss = criterion(E_totals, local_ys)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    E_totals = torch.zeros(batch_size)
                    local_ys = torch.zeros(batch_size)
        saveModel_linear(model, acsf_obj)
    else:
        print("loading model from ", acsf_obj.paths.linear_model)
        model = atomicNet(Gs_num=Gs_num)
        model.load_state_dict(torch.load(acsf_obj.paths.linear_model))

    with torch.no_grad():
        model.eval()
        Es = []
        for n, x in enumerate(xs):
            y = ys[n]
            E_tot = 0
            for atom in x:
                E_i = model(atom)
                E_tot += E_i
            E_tot = E_tot[0]
            # [linear, target, (target - linear)]
            Es.append([E_tot, y, y - E_tot])
    return Es

    # 1. count number elements in each molecule
    # 2. weight associated with each element
    # E = \Sigma_i^N(w_i*N_i)
    # linear regression minimization
    # equivalent to single linear layer
    # gives optimum weights
    # take linear model, and precompute for all molecules and starting point is energy that comes out
    # save and subtract

    return


def atomic_nn(
    xs: [],
    ys: [],
    acsf_obj: acsf_model,
):
    """
    Constructs a new model from a dataset of the structure...
    xs = [np.array[[atomic number, Gs...]]]
    ys = []
    """
    epochs = acsf_obj.nn_props.epochs
    learning_rate = acsf_obj.nn_props.learning_rate
    batch_size = acsf_obj.nn_props.batch_size

    ys = minimize_error(xs, ys, acsf_obj)
    xs, test_xs, ys, test_ys = prepare_data(xs, ys, 0.8)

    Gs_num = xs[0].size()[1] - 1

    model = atomicNet(Gs_num)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("starting training...")
    E_totals = torch.zeros(batch_size)
    local_ys = torch.zeros(batch_size)
    batch = 0
    print("Epoch,\tTraining E,\t Validation E")

    for epoch in range(epochs):
        train_errors = np.zeros((len(xs)))
        train_errors_total = np.zeros((epochs))
        test_errors = np.zeros((len(xs)))
        test_errors_total = np.zeros((epochs))
        for n, x in enumerate(xs):
            # y = ys[n]
            dif = ys[n][2]
            # E_ref = current_y
            # want... y = E_ref_target - E_linear_model
            E_tot = 0
            for atom in x:
                E_i = model(atom)
                E_tot += E_i
            E_tot = E_tot[0]
            E_totals[batch] = E_tot
            local_ys[batch] = dif
            train_errors[n] = criterion(E_tot, dif).item()
            batch += 1
            if batch == batch_size:
                batch = 0
                # loss = criterion(E_tot, y)
                loss = criterion(E_totals, local_ys)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                E_totals = torch.zeros(batch_size)
                local_ys = torch.zeros(batch_size)

        train_e = np.sum(train_errors) / len(train_errors)
        train_errors_total[epoch] = train_e
        with torch.no_grad():
            model.eval()
            for n, x in enumerate(test_xs):
                y = test_ys[n]
                E_tot = 0
                for atom in x:
                    E_i = model(atom)
                    E_tot += E_i
                E_tot = E_tot[0]
                train_errors[n] = criterion(E_tot, dif).item()
            test_e = np.sum(train_errors) / len(train_errors)
            train_errors_total[epoch] = test_e
            for i in train_errors_total:
                if i != 0 and test_e < i:
                    print("\nSaving Model\n")
                    saveModel_ACSF_model(model, acsf_obj, scales=scales)
        print("{:d},\t{:e},\t {:e}".format(epoch + 1, train_e, test_e))

        # if (epoch + 1) % 1 == 0 and loss:
        #     print('\tEpoch Loss:', epoch + 1, loss.item())

    e_x = range(epochs)
    fig = plt.figure(dpi=400)
    plt.plot(train_errors_total, e_x, '-k', label="Train Errors")
    plt.plot(test_errors_total, e_x, '-k', label="Train Errors")
    plt.xlabel('Error')
    plt.ylabel('Epochs')
    plt.legend()
    plt.savefig(acsf_obj.paths.plot_path)
    return
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

    stats_results(differences, percentages, acsf_obj)
    return
