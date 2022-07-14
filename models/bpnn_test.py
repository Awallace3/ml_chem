import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from .structs import acsf_model
import json
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


# class elementNet(torch.nn.Module):
#     """
#     Contains the default element schematic
#     """
#
#     def __init__(self, Gs_num: int):
#         super().__init__()
#         self.el = nn.Sequential(
#             nn.Linear(Gs_num, 200),
#             nn.ReLU(),
#             nn.Linear(200, 100),
#             nn.ReLU(),
#             nn.Linear(100, 1),
#         )


def elementNet(Gs_num):
    """
    Contains the default element schematic
    """
    return nn.Sequential(
        nn.Linear(Gs_num, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 20),
        nn.ReLU(),
        nn.Linear(20, 1),
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
        # self.h = nn.Sequential(
        #     nn.Linear(Gs_num, 100),
        #     nn.Linear(100, 80),
        #     nn.Linear(Gs_num, 1),
        # )
        # self.c = nn.Sequential(
        #     nn.Linear(Gs_num, 2 * Gs_num),
        #     nn.Linear(2 * Gs_num, Gs_num),
        #     nn.Linear(Gs_num, 1),
        # )
        # self.n = nn.Sequential(
        #     nn.Linear(Gs_num, 2 * Gs_num),
        #     nn.Linear(2 * Gs_num, Gs_num),
        #     nn.Linear(Gs_num, 1),
        # )
        # self.o = nn.Sequential(
        #     nn.Linear(Gs_num, 2 * Gs_num),
        #     nn.Linear(2 * Gs_num, Gs_num),
        #     nn.Linear(Gs_num, 1),
        # )
        # self.f = nn.Sequential(
        #     nn.Linear(Gs_num, 2 * Gs_num),
        #     nn.Linear(2 * Gs_num, Gs_num),
        #     nn.Linear(Gs_num, 1),
        # )
        self.el = {1: self.h, 6: self.c, 7: self.n, 8: self.o, 9: self.f}

    def forward(self, x):
        k = int(x[0])
        v = self.el[k]
        v = v(x[1:])
        return v


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
    with open(save_path + "_scale", 'w') as fp:
        for i in scales:
            fp.write(str(i))
            fp.write('\n')
    return


def saveModel(model, path="./results", scales=[1, 1]):
    """
    Saves current model without overwriting previous models
    """
    ms = glob(path + "/t*")
    ms = [i for i in ms if "_" not in i and "." not in i]
    save_path = path + "/t1"
    if len(ms) > 0:
        cnt = [int(i.split("/t")[-1]) for i in ms]
        v = max(cnt) + 1
        save_path = path + "/t%d" % v
        print('model saved to %s' % save_path)
    torch.save(model.state_dict(), save_path)
    with open(save_path + "_scale", 'w') as fp:
        for i in scales:
            fp.write(str(i))
            fp.write('\n')
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


def stats_results_no_obj(differences: [], percentages: []):
    """
    Returns mean differences, mean percentages, and mean absolute error.
    """
    max_d = max(differences)
    print(differences)
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


def read_scales(path):
    with open(path, 'r') as f:
        data = f.readlines()
    data = [float(i.replace("\n", "")) for i in data]
    return data


def test_atomic_nn_no_obj(xs: [],
                          ys: [],
                          model_save_dir="./results",
                          model_name="t2",
                          Gs_num=30,
                          scales=True):
    """
    Tests a model without training. Use the same ACSF parameters as the model
    being used.
    """
    # rescales to have mean=0 and std=1
    # ys, mu, std = rescale_targets(ys)
    # -378.4653
    # mu, std = 0.0, 1.0
    if scales:
        scale_path = "%s/%s_scale" % (model_save_dir, model_name)
        d = read_scales(scale_path)
        mu, std = d[0], d[1]
        scales = [mu, std]

    xs, test_xs, ys, test_ys = prepare_data(xs, ys, 0.8)

    # testing scaling
    # target_scaler = MinMaxScaler()
    # ys = np.array(ys).reshape(-1, 1)
    # target_scaler.fit(ys)
    # ys = target_scaler.transform(ys)
    # ys = [torch.tensor(i, dtype=torch.float32) for i in ys]

    model = atomicNet(Gs_num)
    model.load_state_dict(torch.load("%s/%s" % (model_save_dir, model_name)))

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
            # testing
            E_tot = E_tot[0]
            # E_tot = target_scaler.inverse_transform(E_tot.reshape(1, -1))
            # print(E_tot, y)
            if scales:
                E_tot = E_tot * std + mu
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
    plt.savefig("bpnn_results.png")

    stats_results_no_obj(differences, percentages)
    return


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
    if scales:
        scale_path = "%s_scale" % (acsf_obj.paths.model_path)
        d = read_scales(scale_path)
        mu, std = d[0], d[1]
        scales = [mu, std]

    xs, test_xs, ys, test_ys = prepare_data(xs, ys, 0.8)
    Gs_num = xs[0].size()[1] - 1

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
            if scales:
                E_tot = E_tot * std + mu
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


def atomic_nn_no_object(
    xs: [],
    ys: [],
    model_save_dir="./results",
    epochs=300,
    learning_rate=1e-1,
    batch_size=32,
):
    """
    Constructs a new model from a dataset of the structure...
    xs = [np.array[[atomic number, Gs...]]]
    ys = []
    """
    ys, mu, std = rescale_targets(ys)
    scales = [mu, std]
    # -378.4653
    # mu, std = 0.0, 1.0

    xs, test_xs, ys, test_ys = prepare_data(xs, ys, 0.8)

    # target_scaler = MinMaxScaler()
    # ys = np.array(ys).reshape(-1, 1)
    # target_scaler.fit(ys)
    # ys = target_scaler.transform(ys)
    # ys = [torch.tensor(i, dtype=torch.float32) for i in ys]
    # ys[n] = torch.tensor(ys[n], dtype=torch.float32)
    # test_ys = target_scaler.transform(test_ys)

    Gs_num = xs[0].size()[1] - 1

    model = atomicNet(Gs_num)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("starting training...")
    ten_p = epochs // 10
    print("Printing every %d epochs" % ten_p)
    for epoch in range(epochs):
        for n, x in enumerate(xs):
            y = ys[n]
            E_tot = 0
            for atom in x:
                E_i = model(atom)
                E_tot += E_i
            E_tot = E_tot[0]
            loss = criterion(E_tot, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % ten_p == 0 and loss:
            print(epoch + 1, loss.item())
            # print(E_tot.size(), y.size())
            print("\t", float(E_tot), float(y))

    saveModel(model, path=model_save_dir, scales=scales)

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

            # mean = 0, std = 1
            E_tot = E_tot * std + mu
            y = y * std + mu
            # E_tot = target_scaler.inverse_transform(E_tot)
            # y = y * std + mu

            dif = y - E_tot
            print(E_tot, y)
            differences.append(dif)
            perc = abs(dif) / y
            percentages.append(perc)

    stats_results_no_obj(differences, percentages)
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

    ys, mu, std = rescale_targets(ys)
    scales = [mu, std]
    # -378.4653
    # mu, std = 0.0, 1.0

    xs, test_xs, ys, test_ys = prepare_data(xs, ys, 0.8)

    # target_scaler = MinMaxScaler()
    # ys = np.array(ys).reshape(-1, 1)
    # target_scaler.fit(ys)
    # ys = target_scaler.transform(ys)
    # ys = [torch.tensor(i, dtype=torch.float32) for i in ys]
    # ys[n] = torch.tensor(ys[n], dtype=torch.float32)
    # test_ys = target_scaler.transform(test_ys)

    Gs_num = xs[0].size()[1] - 1

    model = atomicNet(Gs_num)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("starting training...")
    ten_p = epochs // 10
    print("Printing every %d epochs" % ten_p)
    E_totals = torch.zeros(batch_size)
    local_ys = torch.zeros(batch_size)
    print(E_totals)
    batch = 0
    for epoch in range(epochs):
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

        if (epoch + 1) % ten_p == 0 and loss:
            print(epoch + 1, loss.item())
            # print(E_tot.size(), y.size())
            print("\t", float(E_tot), float(y))

    saveModel_ACSF_model(model, acsf_obj, scales=scales)

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

            # mean = 0, std = 1
            E_tot = E_tot * std + mu
            y = y * std + mu
            # E_tot = target_scaler.inverse_transform(E_tot)
            # y = y * std + mu

            dif = y - E_tot
            # print(E_tot, y)
            differences.append(dif)
            perc = abs(dif) / y
            percentages.append(perc)

    stats_results(differences, percentages, acsf_obj)
    return
