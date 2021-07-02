import unittest
import pandas as pd
import ratchet_queue2 as rq


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.df1 = pd.DataFrame({'id': [1,2,3], 'f1': [0.1, 0.3, 0.2],
                           'f2': [0.9, 0.8, 0.1]})

    def test_construction(self):
        rc = rq.RatchetState(self.df1)
        self.assertEqual(len(rc.get_queues()), 2)
        rcq = rc.get_queues()[0]
        q = rcq.get_queue()
        self.assertEqual(len(q), 3)
        self.assertIs(type(q[0]), rq.RatchetNode)
        rn = q[1] # second item
        self.assertAlmostEqual(rn.get_feature(0), 0.2)
        self.assertAlmostEqual(rn.get_feature(1), 0.1)

        rcq = rc.get_queues()[1]
        q = rcq.get_queue()
        self.assertEqual(len(q), 3)
        self.assertIs(type(q[0]), rq.RatchetNode)
        rn = q[1] # second item
        self.assertAlmostEqual(rn.get_feature(0), 0.3)
        self.assertAlmostEqual(rn.get_feature(1), 0.8)

    def test_ratchet(self):
        rc = rq.RatchetState(self.df1)
        s1 = rc.ratchet([0.05, 0.05])
        self.assertEqual(len(s1), 0)
        s2 = rc.ratchet([0.1, 0.05])
        self.assertEqual(len(s2), 0)
        s3 = rc.ratchet([0.2, 0.1])
        self.assertEqual(len(s3), 1)
        s4 = rc.ratchet([0.1, 1.0])
        self.assertEqual(len(s4), 1)
        s5 = rc.ratchet([0.2, 1.0])
        self.assertEqual(len(s5), 2)
        s6 = rc.ratchet([0.2, 0.8])
        self.assertEqual(len(s6), 1)
        s7 = rc.ratchet([1.0, 1.0])
        self.assertEqual(len(s7), 3)

    def test_clank(self):
        rc = rq.RatchetState(self.df1)
        len1 = rc.get_queues()[0].queue_len()
        s1 = rc.ratchet([0.05, 0.05])
        rc.clank(s1)
        len2 = rc.get_queues()[0].queue_len()
        self.assertEqual(len1, len2)

        rc = rq.RatchetColl(self.df1)
        len1 = rc.get_queues()[0].queue_len()
        s1 = rc.ratchet([0.2, 1.0])
        rc.clank(s1)
        len2 = rc.get_queues()[0].queue_len()
        self.assertEqual(len1, len2+2)

if __name__ == '__main__':
    unittest.main()
