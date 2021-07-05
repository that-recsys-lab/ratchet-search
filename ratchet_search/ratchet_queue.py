from __future__ import annotations
import math
from sortedcontainers import SortedKeyList
import pandas as pd
import itertools
import copy
import numpy as np

from typing import Tuple, List

def magnitude(lst, weights=None):
    if weights is None:
        weights = itertools.repeat(1, len(lst))
    sqval = [val * val * wt for val, wt in zip(lst, weights)]
    return math.sqrt(sum(sqval))

class BoundingBox:

    # Can be initialized with a list of nodes, a list of points, or a list of values
    # If it is not a list, it is interpreted to be the number of dimension for minimum box
    def __init__(self, init_data):
        if type(init_data) is list:
            if type(init_data[0]) is RatchetNode:
                interior_points = [list(node.features) for node in init_data]
                self.limits = np.array(map(max, zip(*interior_points)))
            elif type(init_data[0]) is list:
                self.limits = np.array(map(max, zip(*init_data)))
            else:
                self.limits = np.array(init_data)
        elif type(init_data) is np.ndarray:
            self.limits = init_data
        else:
            self.limits = np.full(init_data, 0.0)

    def __repr__(self):
        return f'BoundingBox({self.limits})'

    def __str__(self):
        return self.__repr__()

    def __copy__(self):
        cls = self.__class__
        result: BoundingBox = cls.__new__(cls)
        result.limits = self.limits.copy()
        return result

    # If there are any negative components in the difference, it is outside
    def encloses(self, point: np.ndarray):
        diff = self.limits - point
        neg_count = np.sum(diff < 0)
        return neg_count == 0

    def encloses_node(self, node: RatchetNode):
        return self.encloses(node.features)

    def relax_dim(self, dim):
        new_box = copy.copy(self)
        new_box.limits[dim] = np.inf
        return new_box

    def relax_point(self, point: np.ndarray):
        new_box = copy.copy(self)
        new_box.limits = np.maximum(self.limits, point)
        return new_box

class RatchetNode:

    def __init__(self, id: int, features: tuple):
        self.id = int(id)
        self.features = np.array(features)
        self._mag = np.linalg.norm(self.features)

    def __repr__(self):
        return f'RatchetNode({self.id}, {self.features})'

    def __str__(self):
        return f'<RNode {self.id}: {self.features}>'

    def get_feature(self, idx: int):
        return self.features[idx]

    def get_mag(self):
        return self._mag

    # Might be better to cache this, too. The weights don't change.
    def score(self, weights: np.ndarray):
        return np.sqrt((weights *  self.features * self.features).sum())

    @staticmethod
    def score_list(lst: List[RatchetNode], weights: np.ndarray):
        return sum([node.score(weights) for node in lst])

class RatchetQueue:

    def __init__(self):
        self._queue: SortedKeyList[RatchetNode] = SortedKeyList(key=lambda node: node.get_mag())

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

    def get_next(self, dim, boundary: BoundingBox=None):
        # No boundary is the flag for no filtering
        if boundary is None:
            return self._queue[0]
        else:
            return self.get_next_boundary(boundary.relax_dim(dim))

    def get_next_boundary(self, boundary: BoundingBox):
        filt = filter(lambda node: boundary.encloses_node(node), self._queue)
        try:
            node = next(filt)
        except StopIteration:
            return None
        return node

    def calc_ratchet(self, boundary: BoundingBox):
        # bound_distsq = pow(sum([f*f for f in boundary]), 0.5)
        to_drop = [rn for rn in self._queue if boundary.encloses_node(rn)]
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
        self._weights: np.ndarray = np.array(weights)
        self._boundary: BoundingBox = BoundingBox(len(weights)) # scalar version produces infinite bounds

        for id, row in df.iterrows():
            id = row[0]
            features = list(row[1:])
            self.add(id, tuple(features))

    def __repr__(self):
        return f'<RatchetState() {self._queue.queue_len()} elements>'

    def __copy__(self):
        cls = self.__class__
        result: RatchetState = cls.__new__(cls)
        result._queue = copy.copy(self._queue)
        result._dropped = self._dropped.copy()
        result._weights = self._weights
        result._boundary = copy.copy(self._boundary)
        return result

    def add(self, id, features):
        rn = RatchetNode(id, features)
        self._queue.add(rn)

    def ratchet(self):
        to_drop = self._queue.calc_ratchet(self._boundary)
        return to_drop

    def clank(self, to_drop: List[RatchetNode]):
        self._queue.remove_nodes(to_drop)
        for node in to_drop:
            self._dropped.append(node)

    def filter_node(self, node: RatchetNode):
        features = node.features
        self._boundary = self._boundary.relax_point(features)
        # Ratchet should augment the bounding box not replace it
        to_drop = self.ratchet()
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