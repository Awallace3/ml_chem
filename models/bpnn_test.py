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
import math
import time


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


def elementNet(Gs_num, nodes=[64, 64, 32, 1]):
    """
    Contains the default element schematic
    Gs_num is the size of the input layer
    nodes are the sizes of the hidden layers and last one is output
    """
    return nn.Sequential(
        nn.Linear(Gs_num, nodes[0]),
        nn.ReLU(),
        nn.Linear(nodes[0], nodes[1]),
        nn.ReLU(),
        nn.Linear(nodes[1], nodes[2]),
        nn.ReLU(),
        nn.Linear(nodes[2], nodes[3]),
    )


class atomicNet(torch.nn.Module):
    """
    Constructs NN for H, C, N, O, and F individually
    """

    def __init__(self, Gs_num: int, nodes: []):
        super().__init__()
        self.h = elementNet(Gs_num, nodes)
        self.c = elementNet(Gs_num, nodes)
        self.n = elementNet(Gs_num, nodes)
        self.o = elementNet(Gs_num, nodes)
        self.f = elementNet(Gs_num, nodes)
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
):
    """
    Saves current model
    """

    save_path = acsf_model.paths.model_path
    torch.save(model.state_dict(), save_path)
    return


def stats_epoch(differences: []):
    """
    stats_epoch reports MAE and Max Error for epochs
    """
    max_d = max(differences)
    abs_dif = []
    for i in differences:
        if i < 0:
            abs_dif.append(-i)
        else:
            abs_dif.append(i)
    mae = sum(abs_dif) / len(abs_dif)
    return mae, max_d


def stats_results(differences: [], acsf_obj: acsf_model):
    """
    Returns mean differences, mean, mean absolute error, and max error.
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


def element_subvision(elements: [int] = [1, 6, 7, 8, 9]):
    """
    Generates the element dictionary for value lookups
    """
    el_dc = {}
    for n, i in enumerate(elements):
        el_dc[i] = n
    return el_dc


def minimize_error2(
    xs: [],
    ys: [],
    acsf_obj: acsf_model,
    els: [] = [1, 6, 7, 8, 9],
):
    """
    Minimizes error before neural network training through least squares fitting.
    """
    el_dc = element_subvision()
    b = np.array(ys)
    elements, Gs = np.shape(xs[0])
    Gs -= 1
    Gs_per_el = Gs // len(el_dc)

    print('Number of Molecules:', len(xs))
    A = np.zeros((len(xs), len(el_dc)))
    for i in range(len(xs)):
        m = xs[i]
        for j in range(len(m)):
            e_j = el_dc[m[j, 0]]
            A[i, e_j] += 1

    coef, *v = np.linalg.lstsq(A, b, rcond=None)
    print(coef)
    y_pred = np.zeros(np.shape(b))
    for i in range(len(A)):
        A[i, :] = np.multiply(A[i, :], coef)
        y_pred[i] = np.sum(A[i, :])

    MSE = np.square(np.subtract(b, y_pred)).mean()
    RMSE = math.sqrt(MSE)
    print("\nminimize 2...\n")
    print("RMSE of least squares fitting:", RMSE)
    y_comp = np.column_stack((b, y_pred))
    MAE = np.sum(np.abs(np.subtract(b, y_pred))) / len(b)
    print("MAE of least squares fitting :", MAE)

    # 1. count number elements in each molecule
    # 2. weight associated with each element
    # E = \Sigma_i^N(w_i*N_i)
    # linear regression minimization
    # equivalent to single linear layer
    # gives optimum weights
    # take linear model, and precompute for all molecules and starting point is energy that comes out
    # save and subtract

    return y_comp, coef


def minimize_error(
    xs: [],
    ys: [],
    acsf_obj: acsf_model,
    els: [] = [1, 6, 7, 8, 9],
):
    """
    Minimizes error before neural network training through least squares fitting.
    """
    el_dc = element_subvision()
    b = np.array(ys)
    elements, Gs = np.shape(xs[0])
    Gs -= 1
    Gs_per_el = Gs // len(el_dc)

    print('Number of Molecules:', len(xs))
    A = np.zeros((len(xs), Gs * Gs_per_el))
    for i in range(len(xs)):
        m = xs[i]
        for j in range(len(m)):
            Gs_el = m[j, :]
            e_j = el_dc[m[j, 0]]
            for k in range(len(Gs_el)):
                A[i, e_j * Gs_per_el + k] += Gs_el[k]

    coef, *v = np.linalg.lstsq(A, b, rcond=None)
    y_pred = np.zeros(np.shape(b))
    for i in range(len(A)):
        A[i, :] = np.multiply(A[i, :], coef)
        y_pred[i] = np.sum(A[i, :])

    MSE = np.square(np.subtract(b, y_pred)).mean()
    RMSE = math.sqrt(MSE)
    print("RMSE of least squares fitting:", RMSE)
    y_comp = np.column_stack((b, y_pred))
    MAE = np.sum(np.abs(np.subtract(b, y_pred))) / len(b)
    print("MAE of least squares fitting :", MAE)

    # 1. count number elements in each molecule
    # 2. weight associated with each element
    # E = \Sigma_i^N(w_i*N_i)
    # linear regression minimization
    # equivalent to single linear layer
    # gives optimum weights
    # take linear model, and precompute for all molecules and starting point is energy that comes out
    # save and subtract

    return y_comp


# train_errors_total[epoch], test_errors_total[epoch]


def nn_progress(
    start: int,
    train_error: float,
    test_error: float,
    epochs: int,
    epoch: int,
    mae: float,
    max_e: float,
):
    epoch_time = time.time()
    t_dif = epoch_time - start
    time_of_whole = (t_dif) * epochs / (epoch + 1)
    time_left = (time_of_whole - t_dif) / 60
    time_left /= 60
    print("{:d},\t {:e},\t {:e}\t {:e}\t {:e}\t {:.2f}".format(
        epoch + 1, train_error, test_error, mae, max_e, time_left))
    #if time_left > 60:
    #    time_left /= 60
    #    print("{:d},\t {:e},\t {:e}\t {:e}\t {:e}\t {:.2f} Hours".format(
    #        epoch + 1, train_error, test_error, mae, max_e, time_left))
    #elif time_left > 1:
    #    print("{:d},\t {:e},\t {:e}\t {:e}\t {:e}\t {:.2f} Minutes".format(
    #        epoch + 1, train_error, test_error, mae, max_e, time_left))
    #else:
    #    time_left *= 60
    #    print("{:d},\t {:e},\t {:e}\t {:e}\t {:e}\t {:.2f} Seconds".format(
    #        epoch + 1, train_error, test_error, mae, max_e, time_left))


def eval_linear(
    molecule,
    coefs: np.array,
    el_dc: dict,
):
    """
    eval_linear provides the atom's estimated value
    """
    A = np.zeros(len(molecule))
    for j in range(len(molecule)):
        e_j = el_dc[molecule[j, 0]]
        A[e_j] += 1
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

    # ys1 = minimize_error(xs, ys, acsf_obj)
    ys, coef = minimize_error2(xs, ys, acsf_obj)
    xs, test_xs, ys, test_ys = prepare_data(xs, ys, 0.8)

    Gs_num = xs[0].size()[1] - 1

    model = atomicNet(Gs_num, acsf_obj.nn_props.nodes)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("starting training...")

    E_totals = torch.zeros(batch_size)
    local_ys = torch.zeros(batch_size)
    lowest_validation_error = np.inf
    batch = 0
    print(
        "Epoch,\tTraining E,\t Validation E,\t MAE, \t Max E, \t ET Hrs."
    )
    train_errors = np.zeros((len(xs)))
    train_errors_total = np.zeros((epochs))
    test_errors_total = np.zeros((epochs))
    el_dc = element_subvision()
    start = time.time()
    for epoch in range(epochs):
        for n, x in enumerate(xs):
            y_target = ys[n][0]
            y_linear = ys[n][1]
            y = torch.tensor(y_target - y_linear)
            y = torch.tensor(y_target)
            local_ys[batch] = y
            # want... y = E_ref_target - E_linear_model
            E_tot = 0
            for atom in x:
                E_i = model(atom)
                E_i += coef[el_dc[int(atom[0])]]
                E_tot += E_i
            E_tot = E_tot[0]
            # should this instead be (E_tot + y_linear) compared with E_target?
            # that is the same as E_tot = E_target - y_linear
            E_totals[batch] = E_tot
            train_errors[n] = criterion(E_tot, y).item()
            batch += 1
            if batch == batch_size:
                batch = 0
                loss = criterion(E_totals, local_ys)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                E_totals = torch.zeros(batch_size)
                local_ys = torch.zeros(batch_size)

        train_e = np.sum(train_errors) / len(train_errors)
        train_errors_total[epoch] = train_e

        E_model_test = torch.zeros(len(test_xs))
        target_test = torch.zeros(len(test_xs))
        with torch.no_grad():
            model.eval()
            differences = []
            for n, x in enumerate(test_xs):
                y_target = test_ys[n][0]
                y_linear = test_ys[n][1]
                # y = torch.tensor(y_target - y_linear)
                y = torch.tensor(y_target)
                target_test[n] = y
                # want... y = E_ref_target - E_linear_model
                E_tot = 0
                for atom in x:
                    E_i = model(atom)
                    E_i += coef[el_dc[int(atom[0])]]
                    E_tot += E_i
                E_tot = E_tot[0]
                E_model_test[n] = E_tot
                dif = float(E_tot - y)
                differences.append(dif)
            test_errors_total[epoch] = criterion(E_model_test,
                                                 target_test).item()
            if test_errors_total[epoch] < lowest_validation_error:
                # print("\nSaving Model\n")
                saveModel_ACSF_model(model, acsf_obj)
        mae, max_e = stats_epoch(differences)

        nn_progress(
            start,
            train_errors_total[epoch],
            test_errors_total[epoch],
            epochs,
            epoch,
            mae,
            max_e,
        )

    print("\nTraining time:", time.time() - start)
    e_x = range(epochs)
    fig = plt.figure(dpi=400)
    plt.plot(train_errors_total, e_x, '-k', label="Train Errors")
    plt.plot(test_errors_total, e_x, '-k', label="Train Errors")
    plt.xlabel('Error')
    plt.ylabel('Epochs')
    plt.legend()
    plt.savefig(acsf_obj.paths.plot_path)

    model = atomicNet(Gs_num, acsf_obj.nn_props.nodes)
    model.load_state_dict(torch.load(acsf_obj.paths.model_path))
    with torch.no_grad():
        model.eval()
        differences = []
        percentages = []
        for n, x in enumerate(test_xs):
            y_target = test_ys[n][0]
            y_linear = test_ys[n][1]
            y = torch.tensor(y_target - y_linear)
            # want... y = E_ref_target - E_linear_model
            E_tot = 0
            for atom in x:
                E_i = model(atom)
                E_tot += E_i
            E_tot = E_tot[0]
            dif = float(E_tot - y)
            differences.append(dif)
    stats_results(differences, acsf_obj)
    return
