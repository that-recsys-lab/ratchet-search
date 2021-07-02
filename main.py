from ratchet_search import RatchetSearch
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_file = "test-data2.csv"
    data = pd.read_csv(data_file)
    data['ID'] = data['ID'].astype('int')
    weights = (1.0, 1.0, 1.0)
    list_length = 10
    search = RatchetSearch(data, weights, list_length)

    ans = search.search()
    print("The following nodes will be dropped:")
    for node in ans:
        print(node)

    id = [node.id for node in ans]  
    x1 = [node.features[0] for node in ans]
    x2 = [node.features[1] for node in ans]
    x3 = [node.features[2] for node in ans]

    df = pd.DataFrame({'ID': id, 'X1': x1, 'X2': x2, 'X3': x3})

    df.to_csv("selected.csv", index=False)

    