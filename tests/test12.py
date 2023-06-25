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
        rb.bn_mod.cpt("A")[{}] = [math.sqrt(0.4), math.sqrt(0.6)]
        rb.bn_arg.cpt("A")[{}] = [1.2, 3.4]
        rb.bn_mod.cpt("B")[{'A': 0}] = [math.sqrt(0.5), math.sqrt(0.5)]
        rb.bn_mod.cpt("B")[{'A': 1}] = [math.sqrt(0.34), math.sqrt(0.66)]
        rb.bn_arg.cpt("B")[{'A': 0}] = [0.5, -0.6]
        rb.bn_arg.cpt("B")[{'A': 1}] = [0.9, 0.15]
        rb.bn_mod.cpt("C")[{'B': 0}] = [math.sqrt(0.9), math.sqrt(0.1)]
        rb.bn_mod.cpt("C")[{'B': 1}] = [math.sqrt(0.25), math.sqrt(0.75)]
        rb.bn_arg.cpt("C")[{'B': 0}] = [0.56, 0.67]
        rb.bn_arg.cpt("C")[{'B': 1}] = [23, 32]
        infer_exacte = Inference_Exact(rb)
        infer_exacte.addTarget("B")
        infer_exacte.addTarget("A")
        infer_exacte.addTarget("C")
        infer_exacte.makeInference()
        infer_jt = Inference_JT(rb)
        infer_jt.makeInference()
        self.assertEqual(round(infer_exacte.posterior("A")[0],5), round(infer_jt.posterior("A")[0],5))
        self.assertEqual(round(infer_exacte.posterior("A")[1],5), round(infer_jt.posterior("A")[1],5))
        self.assertEqual(round(infer_exacte.posterior("C")[0],5), round(infer_jt.posterior("C")[0],5))
        self.assertEqual(round(infer_exacte.posterior("C")[1],5), round(infer_jt.posterior("C")[1],5))
        self.assertEqual(round(infer_exacte.posterior("B")[0],5), round(infer_jt.posterior("B")[0],5))
        self.assertEqual(round(infer_exacte.posterior("B")[1],5), round(infer_jt.posterior("B")[1],5))


if __name__ == '__main__':
    unittest.main()




