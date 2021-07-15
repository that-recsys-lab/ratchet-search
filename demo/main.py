from ratchet_search import BinarySpaceSearch
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

def nodes2df (nodes, labels):
    frames = []
    for node in nodes:
        row_dict = {}
        row_dict[labels[0]] = node.id
        for i in range(0, len(node.features)):
            row_dict[labels[i+1]] = node.features[i]
        frames.append(pd.DataFrame(row_dict, index=[0]))
    return pd.concat(frames, ignore_index=True)

def run_3d_demo(data):
    print("Running 3d search with <1, 1, 1> shape")
    dropped = search_3d(data, 20, (1.0, 1.0, 1.0))
    dropped.to_csv("demo-3d-selected.csv", index=False)

    print("Running 3d search with <4, 2, 1> shape")
    dropped = search_3d(data, 20, (4.0, 2.0, 1.0))
    dropped.to_csv("demo-3d-selected-weights.csv", index=False)

def search_3d(data, list_length, shape):
    search = BinarySpaceSearch(data, shape, list_length)
    boundary = search.search()
    print(f"Boundary is {boundary}")
    return nodes2df(search.enclosed, data.columns)

def run_2d_demo(data):
    print("Running 2d search with <1, 1> shape")
    dropped = search_2d(data, 20, (1.0, 1.0))
    dropped.to_csv("demo-2d-selected.csv", index=False)
    print("Running 2d search with <4, 1> shape")
    dropped = search_2d(data, 20, (4.0, 1.0))
    dropped.to_csv("demo-2d-selected-weights.csv", index=False)

def search_2d(data, list_length, shape):
    data2d = data[['ID', 'X1', 'X2']]
    search = BinarySpaceSearch(data2d, shape, list_length)
    boundary = search.search()
    print(f"Boundary is {boundary}")
    df = nodes2df(search.enclosed, ['ID', 'X1', 'X2'])
    return df


if __name__ == '__main__':
    args = read_args()

    data_file = "demo-data.csv"
    data = pd.read_csv(data_file)
    data['ID'] = data['ID'].astype('int')

    if args['threed'] or args['all']:
        run_3d_demo(data)

    if not args['threed']:
        run_2d_demo(data)