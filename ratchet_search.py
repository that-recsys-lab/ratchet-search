from astar import AStar
from ratchet_queue2 import RatchetState, RatchetNode
import pandas as pd
import math
import copy


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
        print(f'heuristic_score {heuristic_score}')
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

        neighbors = []
        for dim in range(len(self.weights)):
            expander = rc.relax_boundary_node(dim)
            if expander is not None:
                new_rs = copy.copy(rc)
                print(f'boundary: {dim}')
                new_rs.filter_node(expander)
                neighbors.append(new_rs)
        # If no other way to expand, go geometric
        if len(neighbors) == 0:
            new_rs = copy.copy(rc)
            expander = rc.relax_boundary_node(-1)
            print(f'boundary: next')
            new_rs.filter_node(expander)
            neighbors.append(new_rs)

        return neighbors

    def search(self):
        path = self.astar(self.initial, None)
        rc_final = list(path)[-1]
        return rc_final.get_dropped()
