from setup import collect_data
import pprint
from models import bpnn_test

pp = pprint.PrettyPrinter(indent=4)


def main():
    xs, ys = collect_data(1000)
    # pp.pprint(ys)
    bpnn_test.atomic_nn(xs, ys)
    return


if __name__ == "__main__":
    main()
