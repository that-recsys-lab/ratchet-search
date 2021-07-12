# ratchet-search
Robin Burke, Ian Burke, Goran Kulyanin

Ratchet search is a special case of A* search seeking to solve a discrete packing problem in n-dimensional space. The goal is to collect k points, densely, such that an (origin-based) bounding box created by the most extreme points in the set contains all of the points and the total cost of the points in minimized. The value of items is a weighted sum of their components in each dimension, but in principle the function can be arbitrary as long as it is non-monotonic in distance from the origin. 

Each state in the search space consists of a current bounding box and the elements that have been included (dropped) and those remaining. The state can be expanded by extending the bounding box in one dimension, dropping the next node encountered in that direction and extending the bounding box to cover it. But (and this is the ratchet part) when that node is dropped, all other nodes within the new bounding box are also dropped, to maintain the compactness criterion. A ratchet may result in too many points being added, and in that case, the node is not feasible and the search looks to expand in a different direction. 

The output of the search is the dimensions of the bounding box, such that exactly k points are contained within it. 

The code makes use of the `AStar` library from Julien Rialland. 

# New idea

Input is a "shape" and the final boundary should be proportional to this shape:

d0, ... dn

f0, ... fn

normalize d so that the shortest side is length 1.

f0/d'0, ... fn/d'n

Each of these ratios should be the same. Calculate variance of ratios. 

The problem is that there isn't a good heuristic for this. 
 
