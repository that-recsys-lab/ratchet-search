# ratchet-search
Robin Burke, Ian Burke, Goran Kulyanin

Ratchet search is a special case of a n-dimensional range query. In this case, we specify an n-dimensional rectangular shape and a number of points, k. The search process returns a boundary such that (a) exactly k points are contained, and (b) the boundary is proportional to the input shape. 

The basic function is a binary search on possible scalings of the input shape. The most expensive operation is testing points for inclusion in the boundary, which is currently brute force, but could be made more efficient through a k-d tree or similar spatial data structure. (Note scipy.spatial.KDTree does not support range queries, yet.)

