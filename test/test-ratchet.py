import unittest
import pandas as pd
from ratchet_search import ratchet_queue as rq
import numpy as np


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.df1 = pd.DataFrame({'id': [1,2,3], 'f1': [0.1, 0.3, 0.2],
                           'f2': [0.9, 0.8, 0.1]})
        # distsq = [0.82, 0.73, 0.05]
        self.df2 = pd.read_csv('test-data.csv')

    def test_construction(self):
        rc = rq.RatchetState(self.df1, (1.0, 1.0, 1.0))
        q = rc._queue
        self.assertEqual(q.queue_len(), 3)
        self.assertIs(type(q._queue[1]), rq.RatchetNode)
        rn = q._queue[1] # second item
        self.assertAlmostEqual(rn.get_feature(0), 0.3)
        self.assertAlmostEqual(rn.get_feature(1), 0.8)


    def test_ratchet(self):
        rc = rq.RatchetState(self.df1, (1.0, 1.0, 1.0))
        rc._boundary = rq.BoundingBox([0.05, 0.05])
        s1 = rc.ratchet()
        self.assertEqual(len(s1), 0)
        rc._boundary = rq.BoundingBox([0.1, 0.05])
        s2 = rc.ratchet()
        self.assertEqual(len(s2), 0)
        rc._boundary = rq.BoundingBox([0.2, 0.1])
        s3 = rc.ratchet()
        self.assertEqual(len(s3), 1)
        rc._boundary = rq.BoundingBox([0.1, 1.0])
        s4 = rc.ratchet()
        self.assertEqual(len(s4), 1)
        rc._boundary = rq.BoundingBox([0.2, 1.0])
        s5 = rc.ratchet()
        self.assertEqual(len(s5), 2)
        rc._boundary = rq.BoundingBox([0.2, 0.8])
        s6 = rc.ratchet()
        self.assertEqual(len(s6), 1)
        rc._boundary = rq.BoundingBox([1.0, 1.0])
        s7 = rc.ratchet()
        self.assertEqual(len(s7), 3)

    def test_clank(self):
        rc = rq.RatchetState(self.df1, (1.0, 1.0, 1.0))
        len1 = rc._queue.queue_len()
        rc._boundary = rq.BoundingBox([0.05, 0.05])
        s1 = rc.ratchet()
        rc.clank(s1)
        len2 = rc._queue.queue_len()
        self.assertEqual(len1, len2)

        rc = rq.RatchetState(self.df1, (1.0, 1.0, 1.0))
        len1 = rc._queue.queue_len()
        rc._boundary = rq.BoundingBox([0.2, 1.0])
        s1 = rc.ratchet()
        rc.clank(s1)
        len2 = rc._queue.queue_len()
        self.assertEqual(len1, len2+2)

    def test_boundary(self):
        rc = rq.RatchetState(self.df2, (1.0, 1.0, 1.0, 1.0))
        rn1 = rc.relax_boundary_node(-1)
        self.assertEqual(rn1.id, 5)
        rc.filter_node(rn1)
        rn2 = rc.relax_boundary_node(0)
        self.assertEqual(rn2.id, 7)
        rc.filter_node(rn2)
        self.assertAlmostEqual(rc._boundary.limits[0], 0.7)
        self.assertAlmostEqual(rc._boundary.limits[3], 0.2)
        rn3 = rc.relax_boundary_node(2)
        self.assertEqual(rn3, None)

    def test_node(self):
        rn = rq.RatchetNode(0, (0.1, 0.2, 0.8))
        bb1 = rq.BoundingBox([0.1, 0.2, 0.8])
        self.assertTrue(bb1.encloses_node(rn))
        bb2 = rq.BoundingBox([0.2, 0.3, 0.4])
        self.assertFalse(bb2.encloses_node(rn))
        score = np.sqrt(0.1*0.1*2 + 0.2*0.2*1 + 0.8*0.8 * 3)
        self.assertAlmostEqual(rn.score(np.array([2, 1, 3])), score)
        self.assertAlmostEqual(rn.get_mag(), np.sqrt(0.01 + 0.04 + 0.64))

    def test_queue(self):
        rl = rq.RatchetQueue()
        rn1 = rq.RatchetNode(1, (0.1, 0.2, 0.8))
        rn2 = rq.RatchetNode(2, (0.1, 0.2, 0.1))
        rn3 = rq.RatchetNode(3, (0.3, 0.2, 0.1))
        rl.add(rn1)
        rl.add(rn2)
        rl.add(rn3)
        node = rl.get_next(0, None)
        self.assertEqual(node.id, 2)
        rl.remove(node)
        bb1 = rq.BoundingBox([0.1, 0.2, 0.1])
        node = rl.get_next(0, bb1)
        self.assertEqual(node.id, 3)
        bb2 = rq.BoundingBox([0.1, 0.2, 0.1])
        node = rl.get_next(2, bb2)
        self.assertEqual(node.id, 1)


if __name__ == '__main__':
    unittest.main()
