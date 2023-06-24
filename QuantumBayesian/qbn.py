import pyAgrum as gum

class QBN:
    """
    Class QBN represents a Quantum Bayesian Network.
    
    """
    def __init__(self):
        """
        Creates and initializes Quantum Bayesian Network.

        Initializes the Bayesian networks for modules and arguments.

        """
        self.bn_mod = gum.BayesNet('Module')
        self.bn_arg = gum.BayesNet('Argument')
        self.nodes = []

    def add(self, varname, nbrmod):
        """
        Adds a new variable to the network.

        Args:
            varname (str): The name of the variable to be added.
            nbrmod (int): The number of modules associated with the variable.

        """
        self.bn_mod.add(varname, nbrmod)
        self.bn_arg.add(varname, nbrmod)
        self.nodes.append(varname)

    def addArc(self, var1, var2):
        """
        Adds a new arc between the given variables to the network.

        Args:
            var1 (str): The name of the first variable.
            var2 (str): The name of the second variable.

        """
        self.bn_mod.addArc(var1, var2)
        self.bn_arg.addArc(var1, var2)

    def module(self, var):
        """
        Returns the conditional probability table associated with the module of the given variable.

        Args:
            var (str): The name of the variable.

        Returns:
            gum.Potential: The conditional probability table (CPT) associated with the module of the variable.

        """
        return self.bn_mod.cpt(var)

    def argument(self, var):
        """
        Returns the conditional probability table associated with the argument of the given variable.

        Args:
            var (str): The name of the variable.

        Returns:
            gum.Potential: The conditional probability table (CPT) associated with the argument of the variable.

        """
        return self.bn_arg.cpt(var)

    def showQBN(self):
        """
        Displays the network.

        Returns:
            gum.BayesNet: The Bayesian network object representing the network.

        """
        return self.bn_mod

    def listNodes(self):
        """
        Returns the list of nodes in the network.

        Returns:
            list [str]: The list of node names in the network.

        """
        return self.nodes

    def verifcpt(self, var):
        """
        Verifies the validity of the conditional probability table (CPT) of the given variable:
        1) Each module value in each row of the table should be positive and not exceed 1.
        2) The sum of their squares should be equal to 1.

        Args:
            var (str): The name of the variable.

        Returns:
            bool: True if the CPT is valid, False otherwise.

        """
        quad = 0
        I = gum.Instantiation(self.bn_mod.cpt(var))
        initial = I.todict()
        r = self.bn_mod.cpt(var)[initial]
        vl = None
        # Checking the first module value
        for key in initial:
            vl = initial[key]
            break
        if r < 0 or r > 1:
            print("Invalid module")
            return False
        quad = quad + r*r
        I.inc()
        # Verifying each cell of the CPT
        while I.todict() != initial:
            dct = I.todict()
            r = self.bn_mod.cpt(var)[dct]
            if r < 0 or r > 1:
                print("Invalid module")
                return False
           
