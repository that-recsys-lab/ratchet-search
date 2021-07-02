from __future__ import annotations
import math
from sortedcontainers import SortedKeyList
import pandas as pd
import itertools
import copy

from typing import Tuple, List


class RatchetNode:

    def __init__(self, id: int, features: tuple):
        self.id = int(id)
        self.features = features

    def __repr__(self):
        return f'RatchetNode({self.id}, {self.features})'

    def __str__(self):
        return f'<RNode {self.id}: {self.features}>'

    def get_feature(self, idx: int):
        return self.features[idx]

    def get_magsq(self):
        return sum([f*f for f in self.features])

    # Returns true if all of the feature values are less than the comparator
    def under(self, comp_features: tuple):
        ans = all([my_val <= comp_val for my_val, comp_val in zip(self.features, comp_features)])
        print(f'My features: {self.features}. Boundary: {comp_features} => {ans}')
        return ans

    def score(self, weights: tuple):
        return sum([val * wt for val, wt in zip(self.features, weights)])

    @staticmethod
    def score_list(lst, weights):
        return sum([node.score(weights) for node in lst])

class RatchetQueue:

    def __init__(self):
        self._queue: SortedKeyList[RatchetNode] = SortedKeyList(key=lambda node: node.get_magsq())

    def __repr__(self):
        return f'<RatchetQueue({len(self._queue)} items>'

    def __str__(self):
        return f'<RQ head: {self._queue[0]}'

    def __copy__(self):
        cls = self.__class__
        result: RatchetQueue = cls.__new__(cls)
        result._queue = self._queue.copy()
        return result

    def add(self, item: RatchetNode):
        self._queue.add(item)

    def remove_nodes(self, nodes):
        for node in nodes:
            self.remove(node)

    def remove(self, node):
        self._queue.remove(node)

    def get_next(self, dim, boundary=None):
        # No boundary is the flag for no filtering
        if boundary is None:
            return self._queue[0]
        else:
            boundary[dim] = math.inf
            print(f'get_next boundary {boundary}')
            return self.get_next_boundary(boundary)

    def get_next_boundary(self, boundary):
        filt = filter(lambda node: node.under(boundary), self._queue)
        try:
            node = next(filt)
        except StopIteration:
            return None
        return node

    def calc_ratchet(self, boundary: List[float,...]):
        bound_distsq = sum([f*f for f in boundary])
        print(f'boundary: {boundary}')
        to_drop = [rn for rn in self._queue.irange_key(max_key=bound_distsq) \
                         if rn.under(boundary)]
        print(f'adding {len(to_drop)}')
        return to_drop

    def get_queue(self):
        return self._queue

    def queue_len(self):
        return len(self._queue)

class RatchetState:

    # Expected data frame configuration is
    # <id, f1, f2, ..., fk>
    # ids must be integers
    # features must be numeric
    def __init__(self, df: pd.DataFrame, weights: tuple):
        self._queue: RatchetQueue = RatchetQueue()
        self._dropped: List[RatchetNode] = []
        self._weights: Tuple[float, ...] = weights
        self._boundary: List[float, ...] = list(itertools.repeat(math.inf, len(weights)))

        for _, row in df.iterrows():
            id = row[0]
            features = row[1:]
            self.add(id, tuple(features))

    def __repr__(self):
        return f'<RatchetColl() {self._queue.queue_len()} elements>'

    def __copy__(self):
        cls = self.__class__
        result: RatchetState = cls.__new__(cls)
        result._queue = copy.copy(self._queue)
        result._dropped = self._dropped.copy()
        result._weights = self._weights
        result._boundary = self._boundary
        return result

    def add(self, id, features):
        rn = RatchetNode(id, features)
        self._queue.add(rn)

    def ratchet(self, boundary: List[float, ...]):
        to_drop = self._queue.calc_ratchet(boundary)
        return to_drop

    def clank(self, to_drop: List[RatchetNode]):
        self._queue.remove_nodes(to_drop)
        for node in to_drop:
            self._dropped.append(node)
        self.update_boundary()

    def update_boundary(self):
        drop_features = [node.features for node in self._dropped]
        self._boundary = list(map(max, zip(drop_features)))

    def filter_node(self, node: RatchetNode):
        print(f'filtering node {node}')
        print(f'boundary: {self._boundary}')
        features = node.features
        to_drop = self.ratchet(list(features))
        self.clank(to_drop)

    def relax_boundary_node(self, dim=-1):
        if dim < 0:
            return self._queue.get_next(dim, None)
        else:
            return self._queue.get_next(dim, self._boundary)

    def next_k_cost(self, k):
        the_queue = self._queue.get_queue()
        return sum([node.score(self._weights) for node in the_queue[0:k]])

    def get_dropped(self):
        return self._dropped

    def score_dropped(self):
        return RatchetNode.score_list(self.get_dropped(), self._weights)