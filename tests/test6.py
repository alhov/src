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
        rb.add("D", 2)
        rb.add("E", 2)
        rb.add("F", 2)
        rb.add("G", 2)
        rb.add("H", 2)
        rb.add("I", 2)
        rb.add("J", 2)

        rb.addArc("A", "B")
        rb.addArc("A", "C")
        rb.addArc("C", "D")
        rb.addArc("B", "D")
        rb.addArc("A", "E")
        rb.addArc("D", "E")
        rb.addArc("B", "G")
        rb.addArc("G", "I")
        rb.addArc("D", "I")
        rb.addArc("A", "J")
        rb.addArc("E", "H")
        rb.addArc("B", "F")
        rb.addArc("F", "D")

        rb.bn_mod.cpt("A")[{}] = [math.sqrt(0.5), math.sqrt(0.5)]

        rb.bn_arg.cpt("A")[{}] = [2.11, 8.15]

        rb.bn_mod.cpt("B")[{'A': 0}] = [math.sqrt(0.97), math.sqrt(0.03)]
        rb.bn_mod.cpt("B")[{'A': 1}] = [math.sqrt(0.84), math.sqrt(0.16)]

        rb.bn_arg.cpt("B")[{'A': 0}] = [0.7, 0.5]
        rb.bn_arg.cpt("B")[{'A': 1}] = [4, 3]

        rb.bn_mod.cpt("J")[{'A': 0}] = [math.sqrt(0.5), math.sqrt(0.5)]
        rb.bn_mod.cpt("J")[{'A': 1}] = [math.sqrt(0.5), math.sqrt(0.5)]

        rb.bn_arg.cpt("J")[{'A': 0}] = [1.2, 2.1]
        rb.bn_arg.cpt("J")[{'A': 1}] = [32, 3.2]

        rb.bn_mod.cpt("C")[{'A': 0}] = [math.sqrt(0.7), math.sqrt(0.3)]
        rb.bn_mod.cpt("C")[{'A': 1}] = [math.sqrt(0.4), math.sqrt(0.6)]

        rb.bn_arg.cpt("C")[{'A': 0}] = [0.3, 0.45]
        rb.bn_arg.cpt("C")[{'A': 1}] = [0.345, 0.897]

        rb.bn_mod.cpt("F")[{'B': 0}] = [math.sqrt(0.75), math.sqrt(0.25)]
        rb.bn_mod.cpt("F")[{'B': 1}] = [math.sqrt(0.2), math.sqrt(0.8)]

        rb.bn_arg.cpt("F")[{'B': 0}] = [1, 2]
        rb.bn_arg.cpt("F")[{'B': 1}] = [11, 12]

        rb.bn_mod.cpt("G")[{'B': 0}] = [math.sqrt(0.1), math.sqrt(0.9)]
        rb.bn_mod.cpt("G")[{'B': 1}] = [math.sqrt(0.9), math.sqrt(0.1)]

        rb.bn_arg.cpt("G")[{'B': 0}] = [1, -1]
        rb.bn_arg.cpt("G")[{'B': 1}] = [-1, 1]

        rb.bn_mod.cpt("H")[{'E': 0}] = [math.sqrt(0.36), math.sqrt(0.64)]
        rb.bn_mod.cpt("H")[{'E': 1}] = [math.sqrt(0.12), math.sqrt(0.88)]

        rb.bn_arg.cpt("H")[{'E': 0}] = [0, 0]
        rb.bn_arg.cpt("H")[{'E': 1}] = [0, 0]

        rb.bn_mod.cpt("E")[{'D': 0, 'A': 0}] = [math.sqrt(0.41), math.sqrt(0.59)]
        rb.bn_mod.cpt("E")[{'D': 1, 'A': 0}] = [math.sqrt(0.35), math.sqrt(0.65)]
        rb.bn_mod.cpt("E")[{'D': 0, 'A': 1}] = [math.sqrt(0.99), math.sqrt(0.01)]
        rb.bn_mod.cpt("E")[{'D': 1, 'A': 1}] = [math.sqrt(0.4), math.sqrt(0.6)]

        rb.bn_arg.cpt("E")[{'D': 0, 'A': 0}] = [4, -5]
        rb.bn_arg.cpt("E")[{'D': 1, 'A': 0}] = [3.123, 0.7]
        rb.bn_arg.cpt("E")[{'D': 0, 'A': 1}] = [0.05, 0.95]
        rb.bn_arg.cpt("E")[{'D': 1, 'A': 1}] = [0.9, 9]

        rb.bn_mod.cpt("I")[{'G': 0, 'D': 0}] = [math.sqrt(0.06), math.sqrt(0.94)]
        rb.bn_mod.cpt("I")[{'G': 1, 'D': 0}] = [math.sqrt(0.5), math.sqrt(0.5)]
        rb.bn_mod.cpt("I")[{'G': 0, 'D': 1}] = [math.sqrt(0.3), math.sqrt(0.7)]
        rb.bn_mod.cpt("I")[{'G': 1, 'D': 1}] = [math.sqrt(0.85), math.sqrt(0.15)]

        rb.bn_arg.cpt("I")[{'G': 0, 'D': 0}] = [1.5, 3]
        rb.bn_arg.cpt("I")[{'G': 1, 'D': 0}] = [4, 2]
        rb.bn_arg.cpt("I")[{'G': 0, 'D': 1}] = [-2, -6]
        rb.bn_arg.cpt("I")[{'G': 1, 'D': 1}] = [0.45, 0.55]

        rb.bn_mod.cpt("D")[{'B': 0, 'F': 0, 'C': 0}] = [math.sqrt(0.54), math.sqrt(0.46)]
        rb.bn_mod.cpt("D")[{'B': 1, 'F': 0, 'C': 0}] = [math.sqrt(0.31), math.sqrt(0.69)]
        rb.bn_mod.cpt("D")[{'B': 0, 'F': 1, 'C': 0}] = [math.sqrt(0.13), math.sqrt(0.87)]
        rb.bn_mod.cpt("D")[{'B': 1, 'F': 1, 'C': 0}] = [math.sqrt(0.07), math.sqrt(0.93)]
        rb.bn_mod.cpt("D")[{'B': 0, 'F': 0, 'C': 1}] = [math.sqrt(0.59), math.sqrt(0.41)]
        rb.bn_mod.cpt("D")[{'B': 1, 'F': 0, 'C': 1}] = [math.sqrt(0.98), math.sqrt(0.02)]
        rb.bn_mod.cpt("D")[{'B': 0, 'F': 1, 'C': 1}] = [math.sqrt(0.5), math.sqrt(0.5)]
        rb.bn_mod.cpt("D")[{'B': 1, 'F': 1, 'C': 1}] = [math.sqrt(0.65), math.sqrt(0.35)]

        rb.bn_arg.cpt("D")[{'B': 0, 'F': 0, 'C': 0}] = [0.06, 0.94]
        rb.bn_arg.cpt("D")[{'B': 1, 'F': 0, 'C': 0}] = [0.045, 0.999]
        rb.bn_arg.cpt("D")[{'B': 0, 'F': 1, 'C': 0}] = [0.233, 0.56]
        rb.bn_arg.cpt("D")[{'B': 1, 'F': 1, 'C': 0}] = [0.235, 0.899]
        rb.bn_arg.cpt("D")[{'B': 0, 'F': 0, 'C': 1}] = [0.999, 0.223]
        rb.bn_arg.cpt("D")[{'B': 1, 'F': 0, 'C': 1}] = [0.8929, 10.020]
        rb.bn_arg.cpt("D")[{'B': 0, 'F': 1, 'C': 1}] = [29, -19]
        rb.bn_arg.cpt("D")[{'B': 1, 'F': 1, 'C': 1}] = [0.2323, 7.2]

        infer_exacte = Inference_Exact(rb)
        infer_exacte.addTarget("B")
        infer_exacte.addTarget("D")
        infer_exacte.addTarget("I")
        infer_exacte.addTarget("H")
        infer_exacte.addTarget("B")
        infer_exacte.addTarget("D")
        infer_exacte.addTarget("I")
        infer_exacte.addTarget("H")
        infer_exacte.addTarget("B")
        infer_exacte.addTarget("D")
        infer_exacte.addTarget("I")
        infer_exacte.addTarget("H")
        infer_exacte.makeInference()
        self.assertEqual(len(infer_exacte.currentTargets()), 4)
        

if __name__ == '__main__':
    unittest.main()
        