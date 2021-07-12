from astar import AStar
from .ratchet_queue import RatchetState, RatchetNode
import pandas as pd
import numpy as np
import math
import copy
from itertools import chain, combinations

def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1))

class RatchetSearch(AStar):

    def __init__(self, data: pd.DataFrame, shape: tuple, goal_count):
        self.data = data
        self.goal_shape = self.normalize_shape(shape)
        self.initial = RatchetState(self.data, self.goal_shape)
        self.goal = goal_count
        self.prior_states_seen = {}

    def normalize_shape(self, shape: tuple):
        shape_fl = [float(val) for val in shape]
        sh_array = np.array(shape_fl)
        return sh_array/np.linalg.norm(sh_array)

    def heuristic_cost_estimate(self, rc: RatchetState, _):
        """Assume additive distortion"""
        #heuristic_score = rc.next_k_cost(self.drops_to_goal(rc))
        #return heuristic_score
        return 0

    def drops_to_goal(self, rs: RatchetState):
        dropped = rs.get_dropped()
        return self.goal - len(dropped)

    def is_goal_reached(self, rc: RatchetState, _):
        current_drop = rc.get_dropped()
        if len(current_drop) == self.goal:
            return True
        else:
            return False

    def distance_between(self, rc1: RatchetState, rc2: RatchetState):
       # If the next node exceeds the goal count, then that is _very_bad_.
       if len(rc2.get_dropped()) > self.goal:
            return math.inf
       # Otherwise the cost is the difference between the scores of the two dropped lists.
       else:
            score1 = rc1.score_dropped()/(self.drops_to_goal(rc1)+1)
            score2 = rc2.score_dropped()/(self.drops_to_goal(rc2)+1)
            return score2 - score1

    def neighbors(self, rc: RatchetState):

        """
        Returns nodes achieved by popping each queue
        """

        expanders = [] # Avoid duplicates
        for dim in range(len(self.goal_shape)):
            expander = rc.relax_boundary_node(dim)
            if expander is not None and expander not in expanders:
                expanders.append(expander)

        exp_combos = powerset(expanders)
        # skip empty set
        next(exp_combos)

        # Note that it is possible for (X,Y) to be redundant with (X) or (Y)
        # To fix this, we would need to check that the bounding box for each
        # (augmented) point set in distinct from all the others. Probably is worth it
        # to trim the search space.
        next_rs_list = []
        for exp_combo in exp_combos:
            new_rs = copy.copy(rc)
            new_rs.update_boundary(exp_combo)
            next_rs_list.append(new_rs)

        # Ensures unique boundaries
        neighbor_dict = {}
        for rs in next_rs_list:
            if rs.get_boundary() not in self.prior_states_seen:
                neighbor_dict[rs.get_boundary()] = rs

        neighbors = []
        for rs in neighbor_dict.values():
            self.prior_states_seen[rs.get_boundary()] = rs

            rs.filter()
            # No need to add nodes that are failures
            if len(rs.get_dropped()) <= self.goal:
                print(f'Increase: {rc} -> {rs}: {self.distance_between(rc, rs)}')
                neighbors.append(rs)

        # If no other way to expand, go geometric
        if len(neighbors) == 0:
            print("No dimensional neighbors")
            new_rs = copy.copy(rc)
            expander = rc.relax_boundary_node(-1)

            if expander not in expanders:
                new_rs.update_boundary([expander])
                self.prior_states_seen[new_rs.get_boundary()] = new_rs
                new_rs.filter()
                neighbors.append(new_rs)

        return neighbors

    def search(self):
        path = self.astar(self.initial, None)
        self.final_state =  list(path)[-1]
        return list(self.final_state._boundary.limits), self.final_state.get_dropped()
