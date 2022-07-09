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
    return 0.5 * math.cos(math.pi * R_ij / R_c) + 1


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

def build_ACSF():
    # symbols = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

    R_c = 6.0
    R_s = 0.5
    eta = 0.5
    acsf = ACSF(
        species=["H", "O", "C", "N", "F"],
        # species=["H", "O"],
        rcut=R_c,
        g2_params=[[0.5, 0.5]],
        # g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
    )
    water = molecule("H2O")
    correct = acsf.create(water, positions=[1])

    h2o_carts = np.array(water.positions)
    h2o_order = np.array(water.numbers)
    # print(h2o_carts, h2o_order)

    G = []
    for i in range(len(h2o_carts[:,0])):
        G_1 = rad_G_1(cut_off_cos, i, h2o_carts, R_c)
        G_2 = rad_G_2(cut_off_cos, i, h2o_carts, R_c, R_s, eta)
        G.append(G_2)
    print(correct)
    print(np.shape(correct))
    print(G)
    print(np.shape(G))

    return


def test():
    np.matrix([0, 0])

    return


def main():
    # data = collect_data(1)
    # print(data)
    build_ACSF()

    return


if __name__ == "__main__":
    main()
