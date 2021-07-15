from ratchet_search import RatchetSearch, RatchetNode, BoundingBox, shape_diff
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
    shape = (1.0, 10.0, 4.0)
    search = RatchetSearch(top_hp, shape, 18)
    boundary, dist, ans = search.search()

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
    print(f'Shape diff: {dist}')

    # best_box = BoundingBox(np.array([0.415, 0.715, 0.230]))
    # search = RatchetSearch(top_hp, (1.0, 1.0, 1.0), 18)
    # best_ans = search.initial._queue.calc_ratchet(best_box)
    #
    # print(f'Boundary: {best_box}')
    # for node in best_ans:
    #     print(f'{node} {node.score(wts)}')
    #
    # tight_box = BoundingBox(best_ans)
    # print(f'Real boundary: {tight_box}')

    # best_box = BoundingBox(boundary)
    # search = RatchetSearch(top_hp, (1.0, 1.0, 1.0), 10)
    # best_ans = search.initial._queue.calc_ratchet(best_box)
    #
    # print(f'Boundary: {best_box}')
    # for node in best_ans:
    #     print(f'{node} {node.score(wts)}')
    #
    # tight_box = BoundingBox(best_ans)
    # print(f'Real boundary: {tight_box}')
