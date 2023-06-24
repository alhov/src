import cmath
import itertools
import math
def produit_complexes(module1, argument1, module2, argument2):
    """
    Computes the product of two complex numbers given their modules and arguments.

    Args:
        module1 (float): The module of the first complex number.
        argument1 (float): The argument of the first complex number in radians.
        module2 (float): The module of the second complex number.
        argument2 (float): The argument of the second complex number in radians.

    Returns:
        complex: The product of the two complex numbers.

    """
    z1 = cmath.rect(module1, argument1)
    z2 = cmath.rect(module2, argument2)
    produit = z1 * z2
    return produit

def normalize_list(lst):
    """
    Normalizes a list of numbers by dividing each element by the sum of all elements.

    Args:
        lst (list): The list of numbers.

    Returns:
        list: The normalized list.

    """
    total_sum = sum(lst)
    normalized_lst = [x / total_sum for x in lst]
    return normalized_lst

def create_complex_number(module, argument):
    """
    Creates a complex number given its module and argument.

    Args:
        module (float): The module of the complex number.
        argument (float): The argument of the complex number in radians.

    Returns:
        complex: The complex number.

    """
    real_part = module * cmath.cos(argument)
    imaginary_part = module * cmath.sin(argument)
    return complex(real_part, imaginary_part)

def complex_to_polar(z):
    """
    Converts a complex number to polar coordinates.

    Args:
        z (complex): The complex number.

    Returns:
        tuple (float, float): A tuple containing the module and argument of the complex number in radians.

    """
    module = abs(z)
    argument = cmath.phase(z)
    return module, argument

def theta_to_exp(bn, cpt):
    """
    Convert the arguments (theta) in a Conditional Probability Table (CPT) to exponentiated form.

    Args:
        bn (gum.BayesNet): The Bayesian network containing the CPT.
        cpt (gum.Potential): The Conditional Probability Table to convert.

    """
    valposib = []
    L = []
    for nd in cpt.names:
        L.append(nd)
        nd_id = bn.variable(nd)
        valposib.append(nd_id.labels())
    for lst in itertools.product(*valposib):
        cpt[{L[i]:lst[i] for i in range(len(L))}] = math.exp(cpt[{L[i]:lst[i] for i in range(len(L))}])
        
def exp_to_theta(bn, cpt):
    """
    Convert the arguments (theta) in a Conditional Probability Table (CPT) from exponentiated form to log-odds.

    Args:
        bn (gum.BayesNet): The Bayesian network containing the CPT.
        cpt (gum.Potential): The Conditional Probability Table to convert.

    """
    valposib = []
    L = []
    for nd in cpt.names:
        L.append(nd)
        nd_id = bn.variable(nd)
        valposib.append(nd_id.labels())
    for lst in itertools.product(*valposib):
        cpt[{L[i]:lst[i] for i in range(len(L))}] = math.log(cpt[{L[i]:lst[i] for i in range(len(L))}])
        
def normalize_cpt(bn, cpt):
    """
    Normalize the probabilities in a Conditional Probability Table (CPT) so that they sum up to 1.

    Args:
        bn (gum.BayesNet): The Bayesian network containing the CPT.
        cpt (gum.Potential): The Conditional Probability Table to normalize.

    """
    valposib = []
    L = []
    for nd in cpt.names:
        L.append(nd)
        nd_id = bn.variable(nd)
        valposib.append(nd_id.labels())
    sumsq = 0
    for lst in itertools.product(*valposib):
        sumsq = sumsq + cpt[{L[i]:lst[i] for i in range(len(L))}] * cpt[{L[i]:lst[i] for i in range(len(L))}]
    delta = 1/sumsq
    for lst in itertools.product(*valposib):
        cpt[{L[i]:lst[i] for i in range(len(L))}] = cpt[{L[i]:lst[i] for i in range(len(L))}] * math.sqrt(delta)