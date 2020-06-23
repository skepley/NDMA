"""
Function and design testing for the HillComponent class

    Output: output
    Other files required: none

    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 4/3/20; Last revision: 6/23/20
"""

import numpy as np
from hill_model import *
from itertools import product


def check_symmetric(tensor):
    """Check if the tensor is symmetric or not"""

    if tensor.ndim == 2:
        return np.max(tensor - tensor.T)

    elif tensor.ndim == 3:
        perms = ['ikj', 'jik', 'jki', 'kji', 'kij']
        return [np.max(tensor - np.einsum('ijk->' + permString, tensor)) for permString in perms]

def check_equal(tensor1, tensor2):
    """check if two tensors are equal up to rounding errors"""
    return np.max(np.abs(tensor1 - tensor2))

# ============= set up an example to test on =============

gamma = 1.2
interactionSign = [1, -1, 1, -1]
interactionType = [2, 1, 1]
interactionIndex = [1, 0, 1, 2, 3]
componentParm = np.array([[1.1, 2.4, 1, 2],
                          [1.2, 2.3, 2, 3],
                          [1.3, 2.2, 3, 2],
                          [1.4, 2.1, 4, 3]], dtype=float)
x = np.array([3., 2, 4, 1], dtype=float)

parameter1 = np.copy(componentParm)
pVars = tuple(zip(*product(range(4), range(4))))  # set all parameters as variable
parameter1[pVars] = np.nan
p = ezcat(gamma, componentParm[pVars])  # get floating points for all variable parameters
f = HillCoordinate(parameter1, interactionSign, interactionType, interactionIndex)
H1 = f.components[0]
H2 = f.components[1]
H3 = f.components[2]
H4 = f.components[3]
pComp = [p[1:5], p[5:9], p[9:13], p[13:]]

# ============= correct derivatives =============
y = f(x, p)  # correct
yx = f.dx(x, p)  # correct
yxx = f.dx2(x, p)  # correct
yp = f.diff(x, p)  # correct
ypx = f.dxdiff(x, p)  # correct

# ============= I think these are correct =============
ypp = f.diff2(x, p)
yxxx = f.dx3(x, p)
ypxx = f.dx2diff(x, p)
yppx = f.dxdiff2(x, p)



stopHere

# ============= check derivatives defined by tensor contraction operations =============
DP = f.diff_interaction(x, p, 1)  # 1-tensor
D2P = f.diff_interaction(x, p, 2)  # 2-tensor
D3P = f.diff_interaction(x, p, 3)  # 3-tensor
DxH = f.diff_component(x, p, [1, 0])  # 2-tensor
DpH = f.diff_component(x, p, [0, 1])  # 2-tensor
DxxH = f.diff_component(x, p, [2, 0])  # 3-tensor
DpxH = f.diff_component(x, p, [1, 1])  # 3-tensor
DppH = f.diff_component(x, p, [0, 2])  # 3-tensor
DppxH = f.diff_component(x, p, [1, 2])  # 4-tensor
DpxxH = f.diff_component(x, p, [2, 1])  # 4-tensor
DxxxH = f.diff_component(x, p, [3, 0])  # 4-tensor

# ============= build all derivatives via tensor contraction operations =============
yx2 = np.einsum('i,ij', DP, DxH)
yx2[f.index] -= gamma  # equal to yx. So DP and DxH are correct
yp2 = ezcat(-x[f.index], np.einsum('i,ij', DP, DpH))  # equal to yp. So DpH is correct
yxx2 = np.einsum('ik,kl,ij', D2P, DxH, DxH) + np.einsum('i,ijk', DP, DxxH)  # equal to yxx. So D2P, DxxH are correct.




# ============= I still don't think these are correct =============

# yppx2 = np.zeros([4, 17, 17])
# term1 = np.einsum('ikq,qr,kl,ij', D3P, DpH, DpH, DxH)
# term2 = np.einsum('ik,kl,ijq', D2P, DpH, DpxH)
# term3 = np.einsum('ik,ij,klq', D2P, DxH, DppH)
# term4 = np.einsum('il,lq,ijk', D2P, DpH, DpxH)
# term5 = np.einsum('i, ijkl', DP, DppxH)
# yppx2[np.ix_(np.arange(4), np.arange(1,17), np.arange(1,17))] = term1 + term2 + term3 + term4 + term5
# print(check_equal(yppx, yppx2))


# # get vectors of appropriate partial derivatives of H (inner terms of chain rule)
# DxH = f.diff_component(x, p, [1, 0], fullTensor=True)
# DxxH = f.diff_component(x, p, [2, 0], fullTensor=True)
# DxxxH = f.diff_component(x, p, [3, 0], fullTensor=True)
#
# # get tensors for derivatives of p o H(x) (outer terms of chain rule)
# Dp = f.diff_interaction(x, p, 1)  # 1-tensor
# D2p = f.diff_interaction(x, p, 2)  # 2-tensor
# D3p = f.diff_interaction(x, p, 3)  # 3-tensor


parameter2 = np.copy(componentParm)
p2Vars = [[0, -1], [1, 0]]  # set n_1, and ell_2 as variable parameters
parameter2[0, -1] = parameter1[1, 0] = np.nan
p2 = np.array([gamma, componentParm[0, -1], componentParm[1, 0]], dtype=float)
f2 = HillCoordinate(parameter2, interactionSign, interactionType, [0, 1, 2])  # gamma is a variable parameter too
print(f2(x, p2))
print(f2.dx(x, p2))
print(f2.dn(x, p2))
print(f2.diff(x, p2, 1))
print(f2.dx2(x, p2))

# check that diff and dn produce equivalent derivatives
parameter3 = np.copy(componentParm)
p3Vars = [[0, -1], [1, -1]]  # set n_1, and n_2 as variable parameters
parameter3[0, -1] = parameter3[1, -1] = np.nan
p3 = np.array([componentParm[0, -1], componentParm[1, -1]], dtype=float)
f3 = HillCoordinate(parameter3, interactionSign, interactionType, [0, 1, 2],
                    gamma=gamma)  # gamma is a variable parameter too
print([f3.diff(x, p3, j) for j in range(f3.nVariableParameter)])

# check summand evaluations
parameter4 = np.repeat(np.nan, 12).reshape(3, 4)
interactionType = [2, 1]
interactionSign = [1, 1, -1]
p4 = np.arange(12)
f4 = HillCoordinate(parameter4, interactionSign, interactionType, [0, 1, 2, 3], gamma=gamma)
print(f4.diff_interaction(x, p4, 1))
print(f4.diff(x, p4))
print(f4.dx2(x, p4))
