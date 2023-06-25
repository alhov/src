import math
import itertools
import random
import cmath
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
from QuantumBayesian import inference
from QuantumBayesian import calcule




class Inference_JT(inference.Inference):
    """

    Class Inference_JT represents inference in Quantum Bayesian Network computed using optimized message passing algorithm.

    Inference_JT(qbn) -> Inference_JT

    Args:
        qbn(QBN): the Quantum Bayesian Network in which we want to compute inference.

    """


    def __init__(self, qbn):
        """
        

        """
        self.bn_mod = qbn.bn_mod
        self.bn_arg = qbn.bn_arg
        self.evidence = {}
        self.dct_temp_evid = {}
        self.pot_mod = {}
        self.pot_arg = {}
        self.nodes = qbn.nodes
        self.jt = None
        self.mess_mod = {}
        self.mess_arg = {}

    def showJt(self):
        """
        Displays the junction tree of the Quantum Bayesian Network.

        """
        gnb.sideBySide(gnb.getJunctionTree(self.bn_mod, withNames=True),
                       gnb.getJunctionTree(self.bn_mod, withNames=False))

    def makeJt(self):
        """
        Constructs the junction tree for the Quantum Bayesian Network.

        """

        jtg = gum.JunctionTreeGenerator()
        self.jt = jtg.junctionTree(self.bn_mod)

    def setPotentials(self):
        """
        Sets the potentials for the nodes in the junction tree.

        """
        for i in self.jt.nodes():
            self.pot_mod[f"{self.jt.clique(i)}"] = 1
            self.pot_arg[f"{self.jt.clique(i)}"] = 0
        for nd in self.nodes:
            lbase = []
            cmax = 0
            for i in self.jt.nodes():
                if self.bn_mod.idFromName(nd) in self.jt.clique(i):
                    cnt = 0
                    for nod in self.bn_mod.cpt(nd).names:
                        if self.pot_mod[f"{self.jt.clique(i)}"] == 1:
                            break
                        if nod in self.pot_mod[f"{self.jt.clique(i)}"].names:
                            cnt = cnt + 1
                    if cnt > cmax:
                        cmax = cnt
                    lbase.append((f"{self.jt.clique(i)}", cnt))
            ltemp = [a for (a, b) in lbase if b == cmax]
            k = random.randint(0, len(ltemp) - 1)
            if self.pot_mod[ltemp[k]] == 1:
                self.pot_mod[ltemp[k]] = 1 * self.bn_mod.cpt(nd)
            else:
                self.pot_mod[ltemp[k]] = self.pot_mod[ltemp[k]] * self.bn_mod.cpt(nd)
            if self.pot_arg[ltemp[k]] == 0:
                self.pot_arg[ltemp[k]] = 1 * self.bn_arg.cpt(nd)
            else:
                calcule.theta_to_exp(self.bn_arg, self.pot_arg[ltemp[k]])
                calcule.theta_to_exp(self.bn_arg, self.bn_arg.cpt(nd))
                self.pot_arg[ltemp[k]] = self.pot_arg[ltemp[k]] * self.bn_arg.cpt(nd)
                calcule.exp_to_theta(self.bn_arg, self.bn_arg.cpt(nd))
                calcule.exp_to_theta(self.bn_arg, self.pot_arg[ltemp[k]])

    def calculMessage(self, clique_send, clique_receive):
        """
        Calculates the message to be sent from a sending clique to a receiving clique.

        Args:
            clique_send (int): Identifier of the sending clique.
            clique_receive (int): Identifier of the receiving clique.

        Returns:
            tuple (gum.Potential, gum.Potential): Tuple containing the message potentials for the model part and argument part.

        """
        clique = []
        separator = []
        for nd in self.nodes:
            if self.bn_mod.idFromName(nd) in self.jt.clique(clique_send):
                clique.append(nd)
            if self.bn_mod.idFromName(nd) in self.jt.separator(clique_send, clique_receive):
                separator.append(nd)
        p = gum.Potential()
        p2 = gum.Potential()
        ltemp = []
        for var in separator:
            var_id = self.bn_mod.variable(var)
            varb = gum.LabelizedVariable(var, var, 0)
            for label in var_id.labels():
                varb.addLabel(label)
            ltemp.append(varb)
        for i in range(len(ltemp)-1, -1, -1):
            p.add(ltemp[i])
            p2.add(ltemp[i])
        others = [el for el in self.pot_mod[f"{self.jt.clique(clique_send)}"].names if el not in separator]
        valposib = []
        for var in others:
            var_id = self.bn_mod.variable(var)
            valposib.append(var_id.labels())
        valsep = []
        for sep in separator:
            sep_id = self.bn_mod.variable(sep)
            valsep.append(sep_id.labels())
        mods = []
        args = []
        for lsep in itertools.product(*valsep):
            summ = calcule.create_complex_number(0,0)
            for lst in itertools.product(*valposib):
                dict_temp={}
                for i in range(len(others)):
                    dict_temp[others[i]] = lst[i]
                for i in range(len(separator)):
                    dict_temp[separator[i]] = lsep[i]
                mod = self.pot_mod[f"{self.jt.clique(clique_send)}"][{nd: dict_temp[nd] for nd in self.pot_mod[f"{self.jt.clique(clique_send)}"].names}]
                arg = self.pot_arg[f"{self.jt.clique(clique_send)}"][{nd: dict_temp[nd] for nd in self.pot_arg[f"{self.jt.clique(clique_send)}"].names}]
                res = calcule.create_complex_number(mod, arg)
                summ = summ + res
            tot = calcule.complex_to_polar(summ)
            mods.append(tot[0])
            args.append(tot[1])
        p.fillWith(mods)
        p2.fillWith(args)
        calcule.normalize_cpt(self.bn_mod, p)
        if self.mess_mod[(clique_send, clique_receive)] == 1:
            self.mess_mod[(clique_send, clique_receive)] = 1 * p
            self.mess_mod[(clique_receive, clique_send)] = 1 * p
        else :
            valprob = []
            L = []
            for nodd in p.names:
                L.append(nodd)
                nodd_id = self.bn_mod.variable(nodd)
                valprob.append(nodd_id.labels())
            for liste in itertools.product(*valprob):
                p[{L[i]:liste[i] for i in range(len(L))}] = p[{L[i]:liste[i] for i in range(len(L))}]/self.mess_mod[(clique_send, clique_receive)][{L[i]:liste[i] for i in range(len(L))}]
        if self.mess_arg[(clique_send, clique_receive)] == 0:
            self.mess_arg[(clique_send, clique_receive)] = 1 * p2
            self.mess_arg[(clique_receive, clique_send)] = 1 * p2
        else :
            valprob = []
            L = []
            for nodd in p2.names:
                L.append(nodd)
                nodd_id = self.bn_arg.variable(nodd)
                valprob.append(nodd_id.labels())
            calcule.theta_to_exp(self.bn_arg, p2)
            calcule.theta_to_exp(self.bn_arg, self.mess_arg[(clique_send, clique_receive)])
            for liste in itertools.product(*valprob):
                p2[{L[i]:liste[i] for i in range(len(L))}] = p2[{L[i]:liste[i] for i in range(len(L))}]/self.mess_arg[(clique_send, clique_receive)][{L[i]:liste[i] for i in range(len(L))}]
            calcule.exp_to_theta(self.bn_arg, p2)
            calcule.exp_to_theta(self.bn_arg, self.mess_arg[(clique_send, clique_receive)])
        return (p,p2)

    def sendMessage(self, clique_send, clique_receive):
        """
        Sends a message from a sending clique to a receiving clique in the junction tree.

        Args:
            clique_send (int): Identifier of the sending clique.
            clique_receive (int): Identifier of the receiving clique.

        """
        if self.pot_mod[f"{self.jt.clique(clique_send)}"] == 1 or self.pot_mod[f"{self.jt.clique(clique_receive)}"] == 1:
            return 
        clique_send_nodes = []
        clique_receive_nodes = []
        separator_nodes = []
        for nd in self.nodes:
            if self.bn_mod.idFromName(nd) in self.jt.clique(clique_send):
                clique_send_nodes.append(nd)
            if self.bn_mod.idFromName(nd) in self.jt.clique(clique_receive):
                clique_receive_nodes.append(nd)
            if self.bn_mod.idFromName(nd) in self.jt.separator(clique_send, clique_receive):
                separator_nodes.append(nd)
        others = [nd for nd in self.pot_mod[f"{self.jt.clique(clique_receive)}"].names if nd not in separator_nodes]
        mess = self.calculMessage(clique_send, clique_receive)
        mess_mod = mess[0]
        mess_arg = mess[1]
        valposib = []
        for var in others:
            var_id = self.bn_mod.variable(var)
            valposib.append(var_id.labels())
        valsep = []
        for sep in separator_nodes:
            sep_id = self.bn_mod.variable(sep)
            valsep.append(sep_id.labels())
        calcule.theta_to_exp(self.bn_arg, self.pot_arg[f"{self.jt.clique(clique_receive)}"])
        calcule.theta_to_exp(self.bn_arg, mess_arg)
        for lsep in itertools.product(*valsep):
            dct_sep={}
            for i in range(len(separator_nodes)):
                dct_sep[separator_nodes[i]] = lsep[i]
            for lst in itertools.product(*valposib):
                dct_rec ={}
                for i in range(len(separator_nodes)):
                    dct_rec[separator_nodes[i]] = lsep[i]
                for i in range(len(others)):
                    dct_rec[others[i]] = lst[i]
                self.pot_mod[f"{self.jt.clique(clique_receive)}"][{nd: dct_rec[nd] for nd in self.pot_mod[f"{self.jt.clique(clique_receive)}"].names}] = self.pot_mod[f"{self.jt.clique(clique_receive)}"][{nd: dct_rec[nd] for nd in self.pot_mod[f"{self.jt.clique(clique_receive)}"].names}] * mess_mod[{nd: dct_sep[nd] for nd in mess_mod.names}] 
                self.pot_arg[f"{self.jt.clique(clique_receive)}"][{nd: dct_rec[nd] for nd in self.pot_arg[f"{self.jt.clique(clique_receive)}"].names}] = self.pot_arg[f"{self.jt.clique(clique_receive)}"][{nd: dct_rec[nd] for nd in self.pot_arg[f"{self.jt.clique(clique_receive)}"].names}] * mess_arg[{nd: dct_sep[nd] for nd in mess_arg.names}]    
        calcule.exp_to_theta(self.bn_arg, self.pot_arg[f"{self.jt.clique(clique_receive)}"])
        calcule.exp_to_theta(self.bn_arg, mess_arg)

    def makeInference(self):
        """
        Performs inference using the optimized message passing algorithm.

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
                    
        self.jt = None
        self.pot_mod = {}
        self.pot_arg = {}
        self.makeJt()
        self.setPotentials()
        k = 0
        for i in self.jt.nodes():
            k = k + 1
        self.mess_mod = {}
        self.mess_arg = {}
        for i in self.jt.nodes():
            for j in self.jt.neighbours(i):
                self.mess_mod[(i,j)] = 1
                self.mess_mod[(j,i)] = 1
        for i in self.jt.nodes():
            for j in self.jt.neighbours(i):
                self.mess_arg[(i,j)] = 0
                self.mess_arg[(j,i)] = 0
        neighbours_send = {}
        neighbours_receive = {}
        for i in self.jt.nodes():
            neighbours_send[i] = []
            neighbours_receive[i] = []
        while k!=0:
            for i in self.jt.nodes():
                if len(neighbours_send[i]) == len(self.jt.neighbours(i)) and len(neighbours_receive[i]) == len(self.jt.neighbours(i)):
                    k = k - 1
                if len(neighbours_receive[i]) == len(self.jt.neighbours(i)) - 1:
                    for j in self.jt.neighbours(i):
                        if j not in neighbours_receive[i]:
                            self.sendMessage(i, j)
                            neighbours_send[i].append(j)
                            neighbours_receive[j].append(i)
                if len(neighbours_receive[i]) == len(self.jt.neighbours(i)):
                    for j in self.jt.neighbours(i):
                        if j not in neighbours_send[i]:
                            self.sendMessage(i, j)
                            neighbours_send[i].append(j)
                            neighbours_receive[j].append(i)

    def posterior(self, var):
        """
        Computes the posterior probability distribution of a variable.

        Args:
            var (str): Name of the variable.

        Returns:
            gum.Potential: The posterior probability distribution.

        """
        for i in self.jt.nodes():
            if self.pot_mod[f"{self.jt.clique(i)}"] ==1:
                    continue
            if self.bn_mod.idFromName(var) in self.jt.clique(i) and var in self.pot_mod[f"{self.jt.clique(i)}"].names:
                var_id = self.bn_mod.variable(var)
                L = [el for el in self.pot_mod[f"{self.jt.clique(i)}"].names if el != var]
                valposib = []
                for el in L:
                    el_id = self.bn_mod.variable(el)
                    valposib.append(el_id.labels())
                restab = []
                for name in var_id.labels():
                    summ = calcule.create_complex_number(0,0)
                    for lst in itertools.product(*valposib):
                        vals = {}
                        vals[var] = name
                        for m in range(len(L)):
                            vals[L[m]] = lst[m]
                        mod = self.pot_mod[f"{self.jt.clique(i)}"][{nd:vals[nd] for nd in self.pot_mod[f"{self.jt.clique(i)}"].names}]
                        arg = self.pot_arg[f"{self.jt.clique(i)}"][{nd:vals[nd] for nd in self.pot_arg[f"{self.jt.clique(i)}"].names}]
                        res = calcule.create_complex_number(mod, arg)
                        summ = summ + res
                    restab.append(abs(summ)*abs(summ))
                varb = gum.LabelizedVariable(var, var, 0)
                for label in var_id.labels():
                    varb.addLabel(label)
                p = gum.Potential().add(varb).fillWith(calcule.normalize_list(restab))
                return p
    
    def currentEvidence(self):
        """
        Get the current evidence in the Quantum Bayesian Network.

        Returns:
            dict: A dictionary containing the evidence variables and their observed values.

        """
        return self.evidence
    
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
