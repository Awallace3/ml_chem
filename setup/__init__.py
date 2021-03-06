import numpy as np
import pandas as pd
from rdkit import Chem
from .progressBar import printProgressBar
import rdkit
from dscribe.descriptors import ACSF
from ase.build import molecule
import math
from dataclasses import dataclass
import pickle
from sklearn.linear_model import LinearRegression
"""
reading data inspired by zatayue/MXMNet
"""

# raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
#            'molnet_publish/qm9.zip')
# raw_url2 = 'https://ndownloader.figshare.com/files/3195404'


def cut_off_cos(R_ij, R_c):
    """
    Cut off function with cosine
    """
    if R_ij > R_c:
        return 0
    # return 0.5 * math.cos(math.pi * R_ij / R_c + 1)
    return 0.5 * (math.cos(math.pi * R_ij / R_c) + 1)


def distance_3d(r1, r2):
    return np.linalg.norm(r1 - r2)


def rad_G_1(cut_off_func, i, carts, R_c):
    """
    first implementation of G_1 ACSF
    """
    G_i_1 = 0
    for j in range(len(carts)):
        if i == j:
            continue
        R_ij = distance_3d(carts[i, :], carts[j, :])
        G_i_1 += cut_off_func(R_ij, R_c)
    return G_i_1


def rad_G_2(cut_off_func, i, carts, R_c, R_s, eta):
    """
    first implementation of G_2 ACSF
    """
    G_i_2 = 0
    for j in range(len(carts)):
        if i == j:
            continue
        R_ij = distance_3d(carts[i, :], carts[j, :])
        G_i_2 += math.exp(-eta * (R_ij - R_s)) * cut_off_func(R_ij, R_c)
    return G_i_2


def build_ACSF_dscribe():
    """
    dscribe playing around zone
    """
    # symbols = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
    R_c = 6.0
    acsf = ACSF(
        # species=["H", "O", "C", "N", "F"],
        # species=["H", "O"],
        species=["H", "O"],
        # species=["H"],
        rcut=R_c,
        # g2_params=[[0.5, 0.5], [0.5, 0.5]],
        # g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
    )
    water = molecule("H2O")
    # water = molecule("H2")
    correct = acsf.create(water, positions=[1])
    print(correct)

    # h2o_carts = np.array(water.positions)
    # h2o_order = np.array(water.numbers)
    return


def read_pickle(fname='data.pickle'):
    with open(fname, 'rb') as handle:
        return pickle.load(handle)


def write_pickle(data, fname='data.pickle'):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def remove_extra_wb(line: str):
    """
    Removes extra whitespace in a string for better splitting.
    """
    line = line.replace("    ",
                        " ").replace("   ", " ").replace("  ", " ").replace(
                            "  ", " ").replace("\n ", "\n")
    return line


def convert_str_carts_np_carts(carts: str):
    """
    This will take Cartesian coordinates as a string and convert it to a numpy
    array.
    """
    carts = remove_extra_wb(carts)
    carts = carts.split("\n")
    if carts[0] == "":
        carts = carts[1:]
    if carts[-1] == "":
        carts = carts[:-1]
    ca = []
    for n, line in enumerate(carts):
        a = line.split()
        row = [float(i) for i in a]
        ca.append(row)
    ca = np.array(ca)
    return ca


def get_3d_angle(r1, r2, r3):
    """
    Acquires angle between three points in Cartesian coordinates.
    """
    r21 = r1 - r2
    r23 = r3 - r2

    cos_angle = np.dot(r21, r23) / (np.linalg.norm(r21) * np.linalg.norm(r23))
    return np.arccos(cos_angle)


def build_ACSF(
        carts: str,
        elements=[1, 6],
        G2_params=[(0.4, 0.2), (0.6, 0.8)],  # (eta, Rs)
        G4_params=[(0.4, 2, 1), (0.6, 2, 1),
                   (0.6, 2, -1)],  # (eta, zeta, lambda)
        Rc=6.0):
    """
    First attempt to build ACSF's
    """
    carts = convert_str_carts_np_carts(carts)
    n = len(carts)
    G1 = np.zeros((n, 1))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r_ij = np.linalg.norm(carts[i, 1:] - carts[j, 1:])
            G_i_1 = cut_off_cos(r_ij, Rc)
            G1[i] += G_i_1

    print(G1)

    G2 = np.zeros((n, len(G2_params)))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for p, (eta, Rs) in enumerate(G2_params):
                r_ij = np.linalg.norm(carts[i, 1:] - carts[j, 1:])
                G_i_2 = math.exp(-eta * (r_ij - Rs)) * cut_off_cos(r_ij, Rc)
                G2[i, p] += G_i_2

    print(G2)

    G4 = np.zeros((n, len(G4_params)))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i == j or i == k or j == k:
                    continue
                for p, (eta, zeta, lam) in enumerate(G4_params):
                    r_ij = np.linalg.norm(carts[i, 1:] - carts[j, 1:])
                    r_ik = np.linalg.norm(carts[i, 1:] - carts[k, 1:])
                    r_jk = np.linalg.norm(carts[j, 1:] - carts[k, 1:])
                    exp_p = -eta * (r_ij**2 + r_ik**2 + r_jk**2)
                    fc_3 = cut_off_cos(r_ij, Rc) * cut_off_cos(
                        r_ik, Rc) * cut_off_cos(r_jk, Rc)
                    t_ijk = get_3d_angle(carts[i, 1:], carts[j, 1:], carts[k,
                                                                           1:])
                    G_i_4 = (1 + lam *
                             math.cos(t_ijk))**zeta * math.exp(exp_p) * fc_3
                    G4[i, p] += G_i_4

    atomic_order = np.zeros((n, 1))
    atomic_order[:, 0] = carts[:, 0]
    G = np.hstack((atomic_order, G1, G2, G4))
    print("final G")
    print(G)
    return G


def element_subvision(elements: [int]):
    """
    Generates the element dictionary for value lookups
    """
    el_dc = {}
    for n, i in enumerate(elements):
        el_dc[i] = n
    return el_dc


def build_G1(n: int, carts: np.array, el_dc: {}, Rc: float):
    """
    Constructs radial ACSF based only on distance
    """
    el_n = len(el_dc)
    G1 = np.zeros((n, 1 * el_n))
    for i in range(n):
        el = el_dc[carts[i, 0]]
        for j in range(n):
            if i == j:
                continue
            r_ij = np.linalg.norm(carts[i, 1:] - carts[j, 1:])
            G_i_1 = cut_off_cos(r_ij, Rc)
            G1[i, el] += G_i_1
    return G1


def build_G2(n: int, carts: np.array, el_dc: {}, Rc: float,
             G2_params: [tuple]):
    """
    Constructs Gaussian radial ACSF with eta and Rc parameters
    """
    el_n = len(el_dc)
    G2 = np.zeros((n, len(G2_params) * el_n))
    for i in range(n):
        el = el_dc[carts[i, 0]]
        for j in range(n):
            if i == j:
                continue
            for p, (eta, Rs) in enumerate(G2_params):
                r_ij = np.linalg.norm(carts[i, 1:] - carts[j, 1:])
                G_i_2 = math.exp(-eta * (r_ij - Rs)) * cut_off_cos(r_ij, Rc)
                G2[i, p + el] += G_i_2
    return G2


def build_G4(n: int, carts: np.array, el_dc: {}, Rc: float,
             G4_params: [tuple]):
    """
    Constructs angular ACSF with eta, zeta, and lambda parameters.
    """

    el_n = len(el_dc)
    G4 = np.zeros((n, len(G4_params) * el_n))
    for i in range(n):
        el = el_dc[carts[i, 0]]
        for j in range(n):
            for k in range(n):
                if i == j or i == k or j == k:
                    continue
                for p, (eta, zeta, lam) in enumerate(G4_params):
                    r_ij = np.linalg.norm(carts[i, 1:] - carts[j, 1:])
                    r_ik = np.linalg.norm(carts[i, 1:] - carts[k, 1:])
                    r_jk = np.linalg.norm(carts[j, 1:] - carts[k, 1:])
                    exp_p = -eta * (r_ij**2 + r_ik**2 + r_jk**2)
                    fc_3 = cut_off_cos(r_ij, Rc) * cut_off_cos(
                        r_ik, Rc) * cut_off_cos(r_jk, Rc)
                    t_ijk = get_3d_angle(carts[i, 1:], carts[j, 1:], carts[k,
                                                                           1:])
                    G_i_4 = (1 + lam *
                             math.cos(t_ijk))**zeta * math.exp(exp_p) * fc_3
                    G4[i, p + el] += G_i_4
    return G4


def build_ACSF_atom_subdivided(
        carts: np.array,
        elements=[1, 6, 7, 8, 9],
        G2_params=[(0.4, 0.2), (0.6, 0.8)],  # (eta, Rs)
        G4_params=[(0.4, 2, 1), (0.6, 2, 1),
                   (0.6, 2, -1)],  # (eta, zeta, lambda)
        Rc=6.0,
        verbose=False):
    """
    Constructs ACSF functions for G_1, G_2, and G_4 from Cartesian coordinates
    that include atomic number in the first column.
    """
    n = len(carts)
    el_dc = element_subvision(elements)

    G1 = build_G1(n, carts, el_dc, Rc)
    G2 = build_G2(n, carts, el_dc, Rc, G2_params)
    G4 = build_G4(n, carts, el_dc, Rc, G4_params)

    atomic_order = np.zeros((n, 1))
    atomic_order[:, 0] = carts[:, 0]
    G = np.hstack((atomic_order, G1, G2, G4))
    if verbose:
        print(el_dc)
        print("G1")
        print(G1)
        print("G2")
        print(G2)
        print("G4")
        print(G4)
        print("final G")
        print(G)

    return G


def element_subvision(elements: [int] = [1, 6, 7, 8, 9]):
    """
    Generates the element dictionary for value lookups
    """
    el_dc = {}
    for n, i in enumerate(elements):
        el_dc[i] = n
    return el_dc


@dataclass
class mol_info:
    name: str
    atom_counts: np.array
    energy: float
    carts: np.array


def create_carts_pkl(path_sdf="data/gdb9.sdf",
                     path_csv="data/gdb9.sdf.csv",
                     path_pkl="data/gdb9.pkl"):
    """
    create_carts_pkl creates pkl of relavent information from gdb9 sdf and csv.
    """
    df = pd.read_csv(path_csv)
    el_dc = element_subvision()
    suppl = Chem.SDMolSupplier(path_sdf, removeHs=False, sanitize=False)
    xs, ys = [], []
    n_mols = len(suppl)
    printProgressBar(0,
                     n_mols,
                     prefix='Progress:',
                     suffix='Complete',
                     length=50)
    ms, atom_cnts, es, carts1 = [], [], [], []
    for i, mol in enumerate(suppl):
        m = mol.GetProp("_Name")
        # print(m, float(df.loc[df['mol_id'] == m]["u0"]))
        e = np.float(df.loc[df['mol_id'] == m]["u0"])
        ys.append(e)

        N = mol.GetNumAtoms()
        pos = suppl.GetItemText(i).split('\n')[4:4 + N]
        atoms = mol.GetAtoms()

        carts = []
        atom_cnt = np.zeros(len(el_dc))

        for n, line in enumerate(pos):
            a = line.split()[:3]
            el = int(atoms[n].GetAtomicNum())
            atom_cnt[el_dc[el]] += 1
            anum = float(el)
            row = [anum, float(a[0]), float(a[1]), float(a[2])]
            carts.append(row)
        arr = np.array(carts)

        # d = mol_info(m, atom_cnt, e, carts)
        # d = [m, atom_cnt, e, carts]
        ms.append(m)
        atom_cnts.append(atom_cnt)
        es.append(e)
        carts1.append(carts)
        printProgressBar(i,
                         n_mols,
                         prefix='Progress:',
                         suffix='Complete',
                         length=50)
    data = [ms, atom_cnts, es, carts1]
    write_pickle(data, path_pkl)
    return data


def collect_data_scratch(
    stop=1,
    path_sdf="data/gdb9.sdf",
    path_csv="data/gdb9.sdf.csv",
    progress=False,
    Rc=6.0,
    G2_params=[(0.4, 0.2), (0.6, 0.8), (0.6, 0.2), (0.8, 0.5)],
    G4_params=[(0.4, 2, 1), (0.6, 2, 1), (0.6, 2, -1)],
):
    """
    Collects downloaded qm9 data and constructs ACSF
    """
    # types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    # symbols = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
    df = pd.read_csv(path_csv)
    suppl = Chem.SDMolSupplier(path_sdf, removeHs=False, sanitize=False)
    xs, ys = [], []
    if stop > len(suppl):
        stop = len(suppl)
    if progress:
        printProgressBar(0,
                         stop,
                         prefix='Progress:',
                         suffix='Complete',
                         length=50)
    for i, mol in enumerate(suppl):
        m = mol.GetProp("_Name")
        e = np.float(df.loc[df['mol_id'] == m]["u298_atom"])
        ys.append(e)

        N = mol.GetNumAtoms()
        pos = suppl.GetItemText(i).split('\n')[4:4 + N]
        atoms = mol.GetAtoms()
        carts = []
        for n, line in enumerate(pos):
            a = line.split()[:3]
            anum = float(atoms[n].GetAtomicNum())
            row = [anum, float(a[0]), float(a[1]), float(a[2])]
            carts.append(row)
        arr = np.array(carts)

        G = build_ACSF_atom_subdivided(arr,
                                       G2_params=G2_params,
                                       G4_params=G4_params,
                                       Rc=Rc)
        xs.append(G)
        if progress:
            printProgressBar(i,
                             stop,
                             prefix='Progress:',
                             suffix='Complete',
                             length=50)
        if stop == i + 1:
            break
    # xs = np.float32(xs)
    # ys = np.array(ys, dtype='float32')
    return xs, ys


def create_atomic_energy_guesses(data):
    X = np.array(data[1])
    y = np.array(data[2])
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    y_pred = model.predict(X)
    mae = np.sum(np.abs(np.subtract(y_pred, y))) / len(y)
    avg = np.sum(np.subtract(y_pred, y)) / len(y)
    # print("avg:", avg)
    # print("MAE:", mae)
    rmse = np.sqrt(np.mean((y_pred - y)**2))
    # print("RMSE:", rmse)
    # print("coefs:", model.coef_)
    y_comp = np.zeros((len(y), 2))
    y_comp[:, 0] = y
    y_comp[:, 1] = y_pred
    return model.coef_, y_comp




def collect_data(
    data: [],
    stop=1,
    progress=False,
    Rc=6.0,
    G2_params=[(0.4, 0.2), (0.6, 0.8), (0.6, 0.2), (0.8, 0.5)],
    G4_params=[(0.4, 2, 1), (0.6, 2, 1), (0.6, 2, -1)],
):
    """
    Collects downloaded qm9 data and constructs ACSF

    data = [ms, np.array(atom_cnts), np.array(es), carts1]
    """
    # types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    # symbols = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

    ys = data[2][:stop]
    coords = data[3]

    xs = []
    if stop > len(coords):
        stop = len(coords)
    if progress:
        printProgressBar(
            0,
            stop,
            prefix='Progress:',
            suffix='Complete',
            length=50,
        )
    for i, carts in enumerate(coords):
        carts = np.array(carts)
        G = build_ACSF_atom_subdivided(
            carts,
            G2_params=G2_params,
            G4_params=G4_params,
            Rc=Rc,
        )
        xs.append(G)
        if progress:
            printProgressBar(
                i,
                stop,
                prefix='Progress:',
                suffix='Complete',
                length=50,
            )
        if stop == i + 1:
            break
    # xs = np.float32(xs)
    # ys = np.array(ys, dtype='float32')
    print()
    print()
    print()
    print(len(xs), len(ys))
    return xs, ys


def h2o_geom():
    carts = """
8  0.000000  0.000000  0.000000
1  0.758602  0.000000  0.504284
1  0.758602  0.000000  -0.504284
    """
    return convert_str_carts_np_carts(carts)


def ch4_geom():
    carts = """
6 	0.0000 	0.0000 	0.0000
1	0.6276 	0.6276 	0.6276
1	0.6276 	-0.6276 	-0.6276
1	-0.6276 	0.6276 	-0.6276
1	-0.6276 	-0.6276 	0.6276
    """
    return convert_str_carts_np_carts(carts)
