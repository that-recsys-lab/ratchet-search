from .ratchet_queue import RatchetState, RatchetNode, enclosing_bounds, BoundingBox
import numpy as np
import pandas as pd


class BinarySpaceSearch:

    def __init__(self, df: pd.DataFrame, shape, goal: int):
        self.shape = np.array(shape)
        self.scale_factor = 1.0
        self.scale_last = 0.0
        self.iteration = 1.0
        self.enclosed = []
        self.nodes = []
        self.goal = goal

        for id, row in df.iterrows():
            id = row[0]
            features = list(row[1:])
            self.nodes.append(RatchetNode(id, tuple(features)))
        self.reshape()

    def update_scale(self, down=True):
        last = self.scale_last
        self.scale_last = self.scale_factor
        if down:
            self.scale_factor = self.scale_factor - 0.5**self.iteration
        else:
            self.scale_factor = self.scale_factor + 0.5**self.iteration

    def reshape(self):
        bounds = RatchetNode.nodes_to_bounds(self.nodes)
        factors = [bounds[i] / self.shape[i] for i in range(0, len(bounds))]
        max_factor = max(factors)
        self.shape = self.shape * max_factor

    def update_enclosed(self):
        old_len = len(self.enclosed)
        bbox = BoundingBox(self.shape * self.scale_factor)
        self.enclosed = []
        for node in self.nodes:
            if bbox.encloses_node(node):
                self.enclosed.append(node)
        return len(self.enclosed) - old_len

    def search(self):
        while True:
            changed = self.update_enclosed()
            if len(self.enclosed) == self.goal:
                return self.shape * self.scale_factor
            elif changed == 0 and self.iteration > 20:
                return None
            elif len(self.enclosed) > self.goal:
                self.update_scale(down=True)
            else:
                self.update_scale(down=False)
            self.iteration += 1



