from __future__ import annotations
import numpy as np

from typing import Tuple, List

def enclosing_bounds(points: List[np.ndarray]):
    arr = np.column_stack(points)
    return np.amax(arr, axis=1)

def shape_diff(arr: np.ndarray, shape: np.ndarray):
    """
    Computes the shape difference
    :param arr: Arbitrary shape
    :param shape: Normalized shape descriptor
    :return: Difference in shape
    """
    arr_mag = np.linalg.norm(arr)
    if arr_mag > 0:
        arr = arr / arr_mag
    diff = arr - shape
    return np.linalg.norm(diff)

class BoundingBox:
    # Can be initialized with a list of nodes, a list of points, or a list of values
    # If it is not a list, it is interpreted to be the number of dimension for minimum box
    def __init__(self, init_data):
        if type(init_data) is list:
            if type(init_data[0]) is RatchetNode:
                interior_points = [list(node.features) for node in init_data]
                self.set_limits(np.array(list(map(max, zip(*interior_points)))))
            elif type(init_data[0]) is list:
                self.set_limits(np.array(map(max, zip(*init_data))))
            else:
                self.set_limits(np.array(init_data))
        elif type(init_data) is np.ndarray:
            self.set_limits(init_data)
        else:
            self.set_limits(np.full(init_data, 0.0))

    def __repr__(self):
        return f'BoundingBox({self.limits})'

    def __str__(self):
        return self.__repr__()

    def __copy__(self):
        cls = self.__class__
        result: BoundingBox = cls.__new__(cls)
        result.set_limits(self.limits.copy())
        return result

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, BoundingBox):
            if len(self.limits) == len(other.limits):
                ans = True
                for i in range(0,len(self.limits)):
                    if self.limits[i] != other.limits[i]:
                        ans = False
                        break
                return ans
        return NotImplemented

    def __hash__(self):
        return hash(self.limits.tostring())

    def set_limits(self, limits: np.ndarray):
        self.limits = limits

    # If there are any negative components in the difference, it is outside
    def encloses(self, point: np.ndarray):
        diff = self.limits - point
        neg_count = np.sum(diff < 0)
        return neg_count == 0

    def encloses_node(self, node: RatchetNode):
        return self.encloses(node.features)


class RatchetNode:
    def __init__(self, id: int, features: tuple):
        self.id = int(id)
        self.features = np.array(features)
        self._score = None

    def __repr__(self):
        return f'RatchetNode({self.id}, {self.features})'

    def __str__(self):
        return f'<RNode {self.id}: {self.features}>'

    def get_feature(self, idx: int):
        return self.features[idx]

    @staticmethod
    def nodes_to_bounds(lst: List[RatchetNode]):
        return enclosing_bounds([node.features for node in lst])
