import numpy as np

from ndma.activation import tanhActivation
from ndma.model.model import Model, ezcat
from ndma.model.restricted_model import RestrictedHillModel
from ndma.examples.EMT_model import def_emt_hill_model

'''
This code is an introduction to building your own model and setting parameters.
At construction, fixed parameters and identified Hill coefficients are set and can't be changed later - a new model 
would have to be defined.

In this code:
- the EMT model is build and computed, in this example the structure of the parameter vector is specified further
- a new model is built and printed, with all parameters being free
- the same model is built with identified Hill coefficients
- the same model is built with some parameters fixed
'''

h = def_emt_hill_model()
print('The EMT model: \n', h)

print('calculate the EMT model for a random choice of parameters and phase space variables\nReminder : EMT has automatic identified Hills')
edgeCounts = [2, 2, 2, 1, 3, 2]
parameterVar = [np.array([[np.nan for j in range(3)] for k in range(nEdge)]) for nEdge in edgeCounts]
gammaValues = np.array([j for j in range(1, 7)])
parmValues = [np.random.rand(*np.shape(parameterVar[node])) for node in range(6)]
x = np.random.rand(6)
p = ezcat(*[ezcat(ezcat(tup[0], tup[1].flatten())) for tup in
            zip(gammaValues, parmValues)])  # this only works when all parameters are variable
hill = 4
print(h(x, hill, p))


gamma = [np.nan, np.nan, np.nan, np.nan] # determines that there are 4 unknowns for 4 equations
p1 = np.array([np.nan, np.nan, np.nan, np.nan], dtype=float) # each "block" stands for the parameters associated with ell, delta, theta, Hill coef
p4 = np.array([[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]], dtype=float)
parameter = [p1, p1, p1, p4]

productionSign = [[1], [-1], [1], [1, -1, -1]]
productionType = [[1], [1], [1], [1, 2]]
productionIndex = [[1], [2], [3], [2, 1, 0]]
g = Model(gamma, parameter, productionSign, productionType, productionIndex)
print('Example model:\n', g)

# test a call
x = np.random.random(4)
pars = np.random.random(28)
index_list = [4,9,14,19,23,27] # computed positions of the Hill coefficients (remember the gamma parameter at every equation)
for i in index_list:
    pars[i] = hill
print('computing the example model: ', g(x, pars))

# the same gamma is used, but the other parameters only have 3 items instead of 4, since the Hill coef is free and set separately at computation
p1 = np.array([[np.nan, np.nan, np.nan]], dtype=float)
p4 = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=float)
parameter_small = [p1, p1, p1, p4]
g_tilde = RestrictedHillModel(gamma, parameter_small, productionSign, productionType, productionIndex)
pars_small = np.delete(pars, index_list, axis=0)
print('Same model with identified Hill coefs = ', g_tilde(x, hill, pars_small))

print('Same model, but with some fixed parameters defined at instantiation')
gamma = [1., 2, 3., 3.3]
p1 = np.array([2., np.nan, np.nan, np.nan], dtype=float)
p4 = np.array([[2.2, np.nan, np.nan, np.nan], [2.3, np.nan, np.nan, np.nan], [2.4, np.nan, np.nan, np.nan]], dtype=float)
parameter = [p1, p1, p1, p4]

g_fixed_pars = Model(gamma, parameter, productionSign, productionType, productionIndex)

# test a call
x = np.random.random(4)
pars = np.random.random(18)
index_list = [2,5,8,11,14,17] # re-computed positions of the Hill coefficients
for i in index_list:
    pars[i] = hill
print('computing the example model: ', g_fixed_pars(x, pars))
print('at different parameters')

"""
We also introduce a way to have a different activation function 
"""
H = tanhActivation(1, ell=3.4, delta=1., theta=2.3)
x1 = 2.
print('tanh =', H(x1))
net_spec = """\nX1 : (X1+X2)(~X3)\nX2 : (X1)\nX3 : (X1)(~X2)"""
A = Model.Model_from_string(net_spec,activationFunction=tanhActivation)

print(A)

x = np.array([4, 3, 2.], dtype=float)

print('evaluation with tanh as activation = ')
gamma = np.array([1,2,3.])
p = [1,.2, 5.,1,.2, 5.,1,.2, 5.,1,.2, 5.,1,.2, 5.,1,.2, 5.]
y = A(x, gamma, p)
print(y)

"""
And an automatic way to have Restricted Hill Models from string
"""
A = Model.Model_from_string(net_spec)
A_restricted = RestrictedHillModel.Model_from_Model(A)
print('Full model :\n', A)
print('Restricted model :\n', A_restricted)

"""
Finally, a Hill restricted model is created from a Model 
(even if the original model had a different activation)
"""
A_restricted = RestrictedHillModel.Model_from_Model(A)