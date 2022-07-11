import numpy as np
from dataclasses import dataclass
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import rdkit
import torch
from dscribe.descriptors import ACSF
from ase.build import molecule
import math
"""
reading data inspired by zatayue/MXMNet
"""

# raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
#            'molnet_publish/qm9.zip')
# raw_url2 = 'https://ndownloader.figshare.com/files/3195404'


@dataclass
class mol_i:
    chem_id: str
    u298_atom: float
    atomic_order: []
    carts: np.ndarray


def collect_data(stop=1,
                 path_sdf="data/gdb9.sdf",
                 path_csv="data/gdb9.sdf.csv"):
    """
    Collects downloaded qm9 data
    """
    # types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    # symbols = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
    df = pd.read_csv(path_csv)
    suppl = Chem.SDMolSupplier(path_sdf, removeHs=False, sanitize=False)
    mols = []
    # for i, mol in enumerate(tqdm(suppl)):
    for i, mol in enumerate(suppl):
        m = mol.GetProp("_Name")
        e = float(df.loc[df['mol_id'] == m]["u298_atom"])
        N = mol.GetNumAtoms()
        pos = suppl.GetItemText(i).split('\n')[4:4 + N]
        atoms = mol.GetAtoms()
        carts, atomic_order = [], []
        for n, line in enumerate(pos):
            a = line.split()[:3]
            atomic_number = atoms[n].GetAtomicNum()
            atomic_order.append(atomic_number)
            row = [float(a[0]), float(a[1]), float(a[2])]
            carts.append(row)
        atomic_order = np.array(atomic_order)
        arr = np.array(carts)
        mols.append(mol_i(m, e, atomic_order, arr))

        if stop == i + 1:
            break
    return mols


def cut_off_cos(R_ij, R_c):
    if R_ij > R_c:
        return 0
    return 0.5 * math.cos(math.pi * R_ij / R_c + 1)


def distance_3d(r1, r2):
    return np.linalg.norm(r1 - r2)


def rad_G_1(cut_off_func, i, carts, R_c):
    G_i_1 = 0
    for j in range(len(carts)):
        if i == j:
            continue
        R_ij = distance_3d(carts[i, :], carts[j, :])
        G_i_1 += cut_off_func(R_ij, R_c)
    return G_i_1


def rad_G_2(cut_off_func, i, carts, R_c, R_s, eta):
    G_i_2 = 0
    for j in range(len(carts)):
        if i == j:
            continue
        R_ij = distance_3d(carts[i, :], carts[j, :])
        G_i_2 += math.exp(-eta * (R_ij - R_s)) * cut_off_func(R_ij, R_c)
    return G_i_2


def build_ACSF_dscribe():
    # symbols = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

    R_c = 6.0
    R_s = 0.5
    eta = 0.5
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

    # G = []
    # for i in range(len(h2o_carts[:, 0])):
    #     G_1 = rad_G_1(cut_off_cos, i, h2o_carts, R_c)
    #     G_2 = rad_G_2(cut_off_cos, i, h2o_carts, R_c, R_s, eta)
    #     G.append(G_2)
    #print(np.shape(correct), np.shape(G))
    # print(correct, G)

    return


def test():
    np.matrix([0, 0])

    return


def remove_extra_wb(line: str):
    line = line.replace("    ",
                        " ").replace("   ", " ").replace("  ", " ").replace(
                            "  ", " ").replace("\n ", "\n")

    return line


def convert_str_carts_np_carts(carts: str):
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


# def rad_G_1(cut_off_func, i, carts, R_c):
#     G_i_1 = 0
#     for j in range(len(carts)):
#         if i == j:
#             continue
#         R_ij = distance_3d(carts[i, :], carts[j, :])
#         G_i_1 += cut_off_func(R_ij, R_c)
#     return G_i_1


def get_3d_angle(r1, r2, r3):
    # ba = a - b
    # bc = c - b
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
    el_dc = {}
    for n, i in enumerate(elements):
        el_dc[i] = n
    return el_dc


def build_ACSF_atom_subdivided(
        carts: str,
        elements=[1, 6, 8],
        G2_params=[(0.4, 0.2), (0.6, 0.8)],  # (eta, Rs)
        G4_params=[(0.4, 2, 1), (0.6, 2, 1),
                   (0.6, 2, -1)],  # (eta, zeta, lambda)
        Rc=6.0,
        verbose=False):
    carts = convert_str_carts_np_carts(carts)
    n = len(carts)
    el_dc = element_subvision(elements)
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


def h2o_geom():
    return """
8  0.000000  0.000000  0.000000
1  0.758602  0.000000  0.504284
1  0.758602  0.000000  -0.504284
"""


def ch4_geom():
    return """
6 	0.0000 	0.0000 	0.0000
1	0.6276 	0.6276 	0.6276
1	0.6276 	-0.6276 	-0.6276
1	-0.6276 	0.6276 	-0.6276
1	-0.6276 	-0.6276 	0.6276
    """


def main():
    data = collect_data(1)
    print(data)

    # ch4 = ch4_geom()
    # G = build_ACSF_atom_subdivided(ch4)
    # print(G)

    return


if __name__ == "__main__":
    main()
