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
        dilemme = QBN()
        dilemme.add("A", 2)
        dilemme.add("B", 2)
        dilemme.addArc("A", "B")
        dilemme.bn_mod.cpt("A")[{}] = [math.sqrt(0.5), math.sqrt(0.5)]
        dilemme.bn_arg.cpt("A")[{}] = [0, 2.8151]
        dilemme.bn_mod.cpt("B")[{'A': 0}] = [math.sqrt(0.97), math.sqrt(0.03)]
        dilemme.bn_mod.cpt("B")[{'A': 1}] = [math.sqrt(0.84), math.sqrt(0.16)]

        infer_jt = Inference_JT(dilemme)
        infer_jt.makeInference()
        self.assertEqual(round(infer_jt.posterior("B")[0], 2), 0.63)
        self.assertEqual(round(infer_jt.posterior("B")[1], 2), 0.37)

if __name__ == '__main__':
    unittest.main()