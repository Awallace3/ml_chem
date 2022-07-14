from dataclasses import dataclass


@dataclass
class paths():
    data_path: str = "data/t0.pickle"
    model_path: str = "results/t0"
    plot_path: str = "plots/t0.png"


@dataclass
class acsf_Gs:
    G2_params: []
    G4_params: []
    Rc: float = 5.0


@dataclass
class results:
    avg_dif: float = 0
    mae: float = 0
    max_d: float = 0


@dataclass
class nn_props:
    nodes: []
    batch_size: int = 1
    epochs: int = 300
    learning_rate: float = 1.0e-1
    batch_size: int = 32


@dataclass
class acsf_model:
    name: str
    num_molecules: int
    acsf_Gs: acsf_Gs
    paths: paths = paths()
    results: results = results()
    nn_props: nn_props = nn_props([30])
