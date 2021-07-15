
from ratchet_search import BinarySpaceSearch
import pandas as pd
import numpy as np
import os


if __name__ == '__main__':
    data = pd.read_csv("hp_ret_pg_probs.csv")
    sorted = data.sort_values('yes_hp_en', ascending=False)
    sorted = sorted.reset_index(drop=True)
    top_hp = sorted[:1125]
    bot_hp = sorted[1125:]

    #min_hp = top_hp['yes_hp_en'].min()
    #top_hp['yes_hp_en'] = top_hp['yes_hp_en'] - min_hp
    #shape = (0.415, 0.715, 0.230)
    #shape = (7, 12, 3.8)
    shape = (8, 14, 4.5)
    #shape = (1.0, 10.0, 4.0)
    search = BinarySpaceSearch(top_hp, shape, 18)
    boundary = search.search()

    # 0.415, 0.715, 0.230
    #boundary[0] += min_hp
    exp_info_file = "results.csv"
    df = pd.DataFrame({'shape_x': shape[0], 'shape_y': shape[1], 'shape_z': shape[2],
                    'bound_x': boundary[0], 'bound_y': boundary[1], 'bound_z': boundary[2]},
                      index=[1])

    df.to_csv(exp_info_file, mode='a', header=not os.path.exists(exp_info_file),
              index=False)

    print(f'Shape: {shape}')
    print(f'Boundary: {boundary}')

