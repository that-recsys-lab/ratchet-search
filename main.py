from ratchet_search import RatchetSearch
import pandas as pd
import argparse

def read_args():
    '''
    Parse command line arguments.
    :return:
    '''
    parser = argparse.ArgumentParser(
        description=
        'The ratchet search library for discrete point packing.',
     )

    parser.add_argument("-t", "--threed", help="Run 3D demo", action="store_true")
    parser.add_argument("-a", "--all", help="Run 2D and 3D demos", action="store_true")

    input_args = parser.parse_args()
    return vars(input_args)


def run_3d_demo(data):
    dropped = search_3d(data, 20, (1.0, 1.0, 1.0))
    dropped.to_csv("selected-3d-1.csv", index=False)

    dropped = search_3d(data, 20, (4.0, 2.0, 1.0))
    dropped.to_csv("selected-3d-2.csv", index=False)


def search_3d(data, list_length, weights):
    search = RatchetSearch(data, weights, list_length)

    boundary, nodes = search.search()
    print(f"Boundary is {boundary}")
    print("The following nodes will be dropped:")
    for node in nodes:
        print(node)

    id = [node.id for node in nodes]
    x1 = [node.features[0] for node in nodes]
    x2 = [node.features[1] for node in nodes]
    x3 = [node.features[2] for node in nodes]

    df = pd.DataFrame({'ID': id, 'X1': x1, 'X2': x2, 'X3': x3})

    return df


def run_2d_demo(data):
    dropped = search_2d(data, 20, (1.0, 1.0))
    dropped.to_csv("selected-2d-1.csv", index=False)
    dropped = search_2d(data, 20, (4.0, 1.0))
    dropped.to_csv("selected-2d-2.csv", index=False)


def search_2d(data, list_length, weights):
    data2d = data[['ID', 'X1', 'X2']]

    search = RatchetSearch(data2d, weights, list_length)

    ans = search.search()
    print("The following nodes will be dropped:")
    for node in ans:
        print(node)

    id = [node.id for node in ans]
    x1 = [node.features[0] for node in ans]
    x2 = [node.features[1] for node in ans]

    df = pd.DataFrame({'ID': id, 'X1': x1, 'X2': x2})

    return df


if __name__ == '__main__':
    args = read_args()

    data_file = "test-data2.csv"
    data = pd.read_csv(data_file)
    data['ID'] = data['ID'].astype('int')

    if args['threed'] or args['all']:
        run_3d_demo(data)
    else:
        run_2d_demo(data)