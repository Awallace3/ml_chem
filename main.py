from setup import collect_data
import pprint
from models import bpnn_test

pp = pprint.PrettyPrinter(indent=4)


def main():
    G2_params = [(0.4, 0.2), (0.6, 0.8), (0.6, 0.2), (0.8, 0.5)]
    G2_params = [(0.4, 0.2), (0.6, 0.2)]
    G4_params = [(0.4, 2, 1), (0.6, 2, 1), (0.6, 2, -1)]
    Rc = 5.0
    xs, ys = collect_data(10000,
                          Rc=Rc,
                          G2_params=G2_params,
                          G4_params=G4_params)
    # pp.pprint(ys)
    bpnn_test.atomic_nn(xs, ys, epochs=200)
    return


if __name__ == "__main__":
    main()
