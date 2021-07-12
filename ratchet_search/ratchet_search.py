from astar import AStar
from .ratchet_queue import RatchetState, RatchetNode
import pandas as pd
import math
import copy
from itertools import chain, combinations

prior_states_seen = {}

def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1))

class RatchetSearch(AStar):

    def __init__(self, data: pd.DataFrame, weights: tuple, goal_count):
        self.data = data
        self.initial = RatchetState(self.data, weights)
        self.weights = weights
        self.goal = goal_count

    def heuristic_cost_estimate(self, rc: RatchetState, _):
        """minimum cost if only lowest scoring items are returned"""
        dropped = rc.get_dropped()
        drops_needed = self.goal - len(dropped)
        heuristic_score = rc.next_k_cost(drops_needed)
        return heuristic_score


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
            score1 = rc1.score_dropped()
            score2 = rc2.score_dropped()
            return score2 - score1

    def neighbors(self, rc: RatchetState):

        """
        Returns nodes achieved by popping each queue
        """

        expanders = [] # Avoid duplicates
        for dim in range(len(self.weights)):
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
            if rs.get_boundary() not in prior_states_seen:
                neighbor_dict[rs.get_boundary()] = rs

        neighbors = []
        for rs in neighbor_dict.values():
            prior_states_seen[rs.get_boundary()] = rs

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
                prior_states_seen[new_rs.get_boundary()] = new_rs
                new_rs.filter()
                neighbors.append(new_rs)

        return neighbors

    def search(self):
        path = self.astar(self.initial, None)
        rc_final = list(path)[-1]
        return list(rc_final._boundary.limits), rc_final.get_dropped()
