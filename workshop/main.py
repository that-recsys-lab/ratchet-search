from ratchet_search import RatchetSearch, BoundingBox
import pandas as pd
import numpy as np

if __name__ == '__main__':
    data = pd.read_csv("hp_ret_pg_probs.csv")
    sorted = data.sort_values('yes_hp_en', ascending=False)
    sorted = sorted.reset_index(drop=True)
    top_hp = sorted[:1125]
    bot_hp = sorted[1125:]
    min_hp = top_hp['yes_hp_en'].min()
    top_hp['yes_hp_en'] = top_hp['yes_hp_en'] - min_hp
    wts = (100.0, 1.0, 50.0)
    search = RatchetSearch(top_hp, wts, 18)
    boundary, ans = search.search()

    # 0.415, 0.715, 0.230
    boundary[0] += min_hp
    print(f'Boundary: {boundary}')
    for node in ans:
        node.features[0] = node.features[0] + min_hp
        print(f'{node} {node.score(wts)}')

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
