from QuantumBayesian import inference
import cmath
import itertools
import pyAgrum as gum
from QuantumBayesian import calcule





class Inference_Exact(inference.Inference): 
    """
    Class Inference_Exact represents exact inference in Quantum Bayesian Network.

    Inference_Exact(qbn) -> Inference_Exact

    Args:
        qbn(QBN): the Quantum Bayesian Network in which we want to compute inference.
        
    """
    def __init__(self, qbn):
        """
        Initializes an instance of the class corresponding to the inference in the given Quantum Bayesian Network.

        Args:
            qbn (QBN): An instance of the QBN class representing the Quantum Bayesian Network.

        """
        self.bn_mod = qbn.bn_mod
        self.bn_arg = qbn.bn_arg
        self.targets = []
        self.evidence = {}
        self.dct_temp_evid = {}
        self.inf_mod = []
        self.inf_arg = []
        self.nodes = qbn.nodes
        
    def calcul(self, vals):
        """
        Calculates the complex sum of probabilities for the given variable values.

        Args:
            vals (dict): A dictionary containing the values of variables.

        Returns:
            tuple (float, float): A tuple containing the module and argument of the complex sum.

        """
        L = self.nodes
        others = [el for el in L if el not in vals]
        valposib = []
        for var in others:
            var_id = self.bn_mod.variable(var)
            valposib.append(var_id.labels())
        summ = calcule.create_complex_number(0, 0)
        for lst in itertools.product(*valposib):
            dict_temp = {}
            for i in range(len(others)):
                dict_temp[others[i]] = lst[i]
            for key in vals:
                dict_temp[key] = vals[key]
            mod = 1
            arg = 0
            for el in L:
                mod = mod * self.bn_mod.cpt(el)[{nd: dict_temp[nd] for nd in self.bn_mod.cpt(el).names}]
                arg = arg + self.bn_arg.cpt(el)[{nd: dict_temp[nd] for nd in self.bn_arg.cpt(el).names}]
            res = calcule.create_complex_number(mod, arg)
            summ = summ + res
        return calcule.complex_to_polar(summ)
    
   
    def makeInference(self):
        """
        Performs the inference in the Quantum Bayesian Network based on the given evidence.

        """
        # Restore the original probabilities
        for key in self.dct_temp_evid:
            visited = []
            I = gum.Instantiation(self.bn_mod.cpt(key))
            
            # Iterate over possible values of the variable
            while True:
                dc = I.todict()
                h = ""
                for m in dc:
                    h = dc[m]
                    break
                
                # Check if the value has already been visited
                if h in visited:
                    break
                else:
                    visited.append(h)
                    I.inc()
            
            # Restore the original probabilities
            for v in visited:
                self.bn_mod.cpt(key)[{key: v}] = self.dct_temp_evid[key][v]
        
        self.dct_temp_evid = {}
        
        # Store the original probabilities affected by evidence in dct
        for (key, val) in self.evidence.items():
            if key not in self.dct_temp_evid:
                self.dct_temp_evid[key] = {}
            
            visited = []
            I = gum.Instantiation(self.bn_mod.cpt(key))
            
            # Iterate over possible values of the variable
            while True:
                dc = I.todict()
                h = ""
                for m in dc:
                    h = dc[m]
                    break
                
                # Check if the value has already been visited
                if h in visited:
                    break
                else:
                    visited.append(h)
                    I.inc()
            
            # Store the original probability and set the observed value to 1
            self.dct_temp_evid[key][str(val)] = self.bn_mod.cpt(key)[{key: val}]
            self.bn_mod.cpt(key)[{key: val}] = 1
            
            # Set other values to 0
            for v in visited:
                if v != val and int(v) != val:
                    self.dct_temp_evid[key][v] = self.bn_mod.cpt(key)[{key: v}]
                    self.bn_mod.cpt(key)[{key: v}] = 0
                    
        self.inf_mod = []
        self.inf_arg = []
        
        valposib = []
        for var in self.targets:
            var_id = self.bn_mod.variable(var)
            valposib.append(var_id.labels())
        for lst in itertools.product(*valposib):
            vals = {}
            for i in range(len(self.targets)):
                vals[self.targets[i]] = lst[i]
            h = self.calcul(vals)
            t1 = (vals, h[0])
            t2 = (vals, h[1])
            self.inf_mod.append(t1)
            self.inf_arg.append(t2)
            
    def posterior(self, var):
        """
        Computes the posterior probability distribution of a variable.

        Args:
            var (str): The name of the variable.

        Returns:
            gum.Potential: The posterior probability distribution.

        """
        var_id = self.bn_mod.variable(var)
        L = [el for el in self.targets if el != var]
        valposib = []
        for el in L:
            el_id = self.bn_mod.variable(el)
            valposib.append(el_id.labels())
        restab = []
        for name in var_id.labels():
            summ = calcule.create_complex_number(0, 0)
            for lst in itertools.product(*valposib):
                vals = {}
                vals[var] = name
                for i in range(len(L)):
                    vals[L[i]] = lst[i]
                for el in self.inf_mod:
                    if el[0] == vals:
                        mod = el[1]
                for el in self.inf_arg:
                    if el[0] == vals:
                        arg = el[1]
                res = calcule.create_complex_number(mod, arg)
                summ = summ + res
            restab.append(abs(summ) * abs(summ))
        varb = gum.LabelizedVariable(var, var, 0)
        for label in var_id.labels():
            varb.addLabel(label)
        
        p = gum.Potential().add(varb).fillWith(calcule.normalize_list(restab))
        return p
        
    def posteriorJoint(self, variables):
        """
        Computes the joint posterior probability distribution of a set of variables.

        Args:
            variables (list[str]): The names of the variables.

        Returns:
            gum.Potential: The joint posterior probability distribution.

        """
        others = [var for var in self.targets if var not in variables]
        valposib = []
        valparam = []
        for var in variables:
            var_id = self.bn_mod.variable(var)
            valparam.append(var_id.labels())
        for var in others:
            var_id = self.bn_mod.variable(var)
            valposib.append(var_id.labels())
        restab = []
        for lval in itertools.product(*valparam):
            summ = calcule.create_complex_number(0, 0)
            for lst in itertools.product(*valposib):
                vals = {}
                for i in range(len(variables)):
                    vals[variables[i]] = lval[i]
                for i in range(len(others)):
                    vals[others[i]] = lst[i]
                for el in self.inf_mod:
                    if el[0] == vals:
                        mod = el[1]
                for el in self.inf_arg:
                    if el[0] == vals:
                        arg = el[1]
                res = calcule.create_complex_number(mod, arg)
                summ = summ + res
            restab.append(abs(summ) * abs(summ))
        p = gum.Potential()
        ltemp = []
        for var in variables:
            varb = gum.LabelizedVariable(var, var, 0)
            for label in var_id.labels():
                varb.addLabel(label)
            ltemp.append(varb)
        for i in range(len(ltemp)-1, -1, -1):
            p.add(ltemp[i])
        p.fillWith(calcule.normalize_list(restab))
        return p
    
    def currentTargets(self):
        """
        Get the current list of target variables.

        Returns:
            list [str]: The list of target variables.

        """
        return self.targets
    
    def currentEvidence(self):
        """
        Get the current evidence in the Quantum Bayesian Network.

        Returns:
            dict: A dictionary containing the evidence variables and their observed values.

        """
        return self.evidence
   
    def addTarget(self, var):
        """
        Adds a target variable to the list of targets.

        Args:
            var (str): The name of the target variable to add.

        """
        self.targets.append(var)
        
    
    def addEvidence(self, var, val):
        """
        Adds evidence to the Quantum Bayesian Network by specifying the observed value for a variable.

        Args:
            var (str): The name of the variable for which evidence is provided.
            val: The observed value for the variable.

        """
        self.evidence[var] = val
        
    def removeEvidence(self, var):
        """
        Removes the evidence for a specific variable from the Quantum Bayesian Network.

        Args:
            var (str): The name of the variable for which to remove the evidence.

        """
        del self.evidence[var]
        
    def removeTarget(self, var):
        """
        Removes a target variable from the list of targets.

        Args:
            var (str): The name of the target variable to remove.

        """
        self.targets.remove(var)
