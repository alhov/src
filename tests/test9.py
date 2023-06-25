import os
import sys
import math
import unittest
sys.path.insert(0, os.path.abspath('../QuantumBayesian'))
sys.path.insert(0, os.path.abspath('..'))

from qbn import QBN
from inference_exact import Inference_Exact
from inference_jt import Inference_JT



class MyTest(unittest.TestCase):
    def test_unit(self):
        rb = QBN()
        rb.add("A", 2)
        rb.add("B", 2)
        rb.add("C", 2)
        rb.addArc("A", "B")
        rb.addArc("B", "C")
        rb.addArc("A", "B")
        rb.addArc("B", "C")
        rb.addArc("A", "B")
        rb.addArc("B", "C")
        rb.addArc("A", "B")
        rb.addArc("B", "C")
        

        
    
        self.assertEqual(len(rb.listArcs()), 2)
        
        

if __name__ == '__main__':
    unittest.main()
        