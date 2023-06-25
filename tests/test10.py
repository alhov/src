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
        rb.bn_mod.cpt("A")[{}] = [math.sqrt(0.5), math.sqrt(0.5)]
        rb.bn_arg.cpt("A")[{}] = [0, 2.8151]
        rb.bn_mod.cpt("B")[{'A': 0}] = [math.sqrt(0.97), math.sqrt(0.03)]
        rb.bn_mod.cpt("B")[{'A': 1}] = [math.sqrt(0.84), math.sqrt(0.16)]
        rb.bn_mod.cpt("C")[{'B': 0}] = [math.sqrt(0.7), math.sqrt(0.3)]
        rb.bn_mod.cpt("C")[{'B': 1}] = [math.sqrt(0.4), math.sqrt(0.6)]
        infer_exacte = Inference_Exact(rb)
        infer_exacte.addTarget("B")
        infer_exacte.makeInference()
        infer_jt = Inference_JT(rb)
        infer_jt.makeInference()
        self.assertEqual(round(infer_exacte.posterior("B")[0],5), round(infer_jt.posterior("B")[0],5))
        self.assertEqual(round(infer_exacte.posterior("B")[1],5), round(infer_jt.posterior("B")[1],5))


if __name__ == '__main__':
    unittest.main()




