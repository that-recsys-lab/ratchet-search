from .astar import AStar
from .ratchet_queue import RatchetState, RatchetNode, enclosing_bounds
import pandas as pd
import numpy as np
import math
import copy
from itertools import chain, combinations

def powerset(iterable, skip_empty=False):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    iter = chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1))
    if skip_empty:
        next(iter)
    return iter

class RatchetSearch(AStar):

    def __init__(self, data: pd.DataFrame, shape: tuple, goal_count):
        self.data = data
        self.goal_shape = RatchetSearch.normalize_shape(shape)
        self.initial = RatchetState(self.data, self.goal_shape)
        self.goal = goal_count
        self.prior_states_seen = {}

    @staticmethod
    def normalize_shape(shape: tuple):
        shape_fl = [float(val) for val in shape]
        sh_array = np.array(shape_fl)
        return sh_array/np.linalg.norm(sh_array)

    def heuristic_cost_estimate(self, rc: RatchetState, _):
        """Assume additive distortion"""
        #heuristic_score = self.distance_of(rc)
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

    def distance_of(self, rc: RatchetState):
       # If the next node exceeds the goal count, then that is _very_bad_.
       if len(rc.get_dropped()) > self.goal:
            return math.inf
       # Otherwise the cost is the score of the dropped list.
       else:
            return rc.score_dropped()

    def neighbors(self, rs:RatchetState):

        # An expander is a point that is the closest obtained by
        # relaxing some dimension or combination of dimensions of
        # the current boundary
        expanders = [] # Avoid duplicates
        dims = range(0, len(self.goal_shape))
        dim_combos = powerset(dims, skip_empty=True)
        for dim_combo in dim_combos:
            expander = rs.relax_boundary_node(dim_combo)
            if expander is not None and expander not in expanders:
                expanders.append(expander)

        # We try adding all combinations of points from the set of
        # expanders.
        exp_combos = powerset(expanders, skip_empty=True)
        next_rs_list = []
        for exp_combo in exp_combos:
            new_rs = copy.copy(rs)
            new_rs.update_boundary(exp_combo)
            next_rs_list.append(new_rs)

        # Ensures unique boundaries and drops redundant states
        neighbor_dict = {}
        for next_rs in next_rs_list:
            if next_rs.get_boundary() not in self.prior_states_seen:
                neighbor_dict[next_rs.get_boundary()] = next_rs

        # Collect neighbors. Filter before enqueuing
        neighbors = []
        for next_rs in neighbor_dict.values():
            self.prior_states_seen[next_rs.get_boundary()] = next_rs

            next_rs.filter()
            # No need to add nodes that are failures
            if len(next_rs.get_dropped()) <= self.goal:
                print(f'Next node: {rs} -> {next_rs}: Dist: {self.distance_of(next_rs)}')
                print(f'\tBound: {next_rs.get_boundary()} To drop: {self.drops_to_goal(next_rs)}')
                neighbors.append(next_rs)

        return neighbors

    # def neighbors(self, rc: RatchetState):
    #
    #     """
    #     Returns nodes achieved by popping each queue
    #     """
    #
    #     expanders = [] # Avoid duplicates
    #     for dim in range(len(self.goal_shape)):
    #         expander = rc.relax_boundary_node([dim])
    #         if expander is not None and expander not in expanders:
    #             expanders.append(expander)
    #
    #     exp_combos = powerset(expanders, skip_empty=True)
    #
    #     # Note that it is possible for (X,Y) to be redundant with (X) or (Y)
    #     # To fix this, we would need to check that the bounding box for each
    #     # (augmented) point set in distinct from all the others. Probably is worth it
    #     # to trim the search space.
    #     next_rs_list = []
    #     for exp_combo in exp_combos:
    #         new_rs = copy.copy(rc)
    #         new_rs.update_boundary(exp_combo)
    #         next_rs_list.append(new_rs)
    #
    #     # Ensures unique boundaries
    #     neighbor_dict = {}
    #     for rs in next_rs_list:
    #         if rs.get_boundary() not in self.prior_states_seen:
    #             neighbor_dict[rs.get_boundary()] = rs
    #
    #     neighbors = []
    #     for rs in neighbor_dict.values():
    #         self.prior_states_seen[rs.get_boundary()] = rs
    #
    #         rs.filter()
    #         # No need to add nodes that are failures
    #         if len(rs.get_dropped()) <= self.goal:
    #             print(f'Increase: {rc} -> {rs}: {self.distance_between(rc, rs)}')
    #             neighbors.append(rs)
    #
    #     # If no other way to expand, go geometric
    #     #if len(neighbors) == 0:
    #     #    print("No dimensional neighbors")
    #     new_rs = copy.copy(rc)
    #     expander = rc.relax_boundary_node(-1)
    #
    #     if expander not in expanders:
    #         new_rs.update_boundary([expander])
    #         self.prior_states_seen[new_rs.get_boundary()] = new_rs
    #         new_rs.filter()
    #         neighbors.append(new_rs)
    #
    #     return neighbors

    def search(self):
        path = self.astar(self.initial, None)
        self.final_state =  list(path)[-1]
        return list(self.final_state._boundary.limits), \
                self.final_state.score_dropped(), \
               self.final_state.get_dropped()
