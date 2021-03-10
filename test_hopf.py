import sys

sys.path.insert(0, './models')

from repressilator import Repressilator
import numpy as np
from scipy.integrate import *
import matplotlib.pyplot as plt

#
# "Parameter":{"L[x1->x2]" : 0.9132235573212335, "L[x2->x3]" : 0.08856225631514987, "L[x3->x1]" : 0.081866957498757928,
# "T[x1->x2]" : 0.14056968648960766, "T[x2->x3]" : 0.97137585771026835, "T[x3->x1]" : 0.22193859302282001,
# "U[x1->x2]" : 1.4808731812451179, "U[x2->x3]" : 0.40910071361399719, "U[x3->x1]" : 2.9182373984706249}

L = np.array([0.9132235573212335, 0.08856225631514987, 0.081866957498757928])
T = np.array([0.14056968648960766, 0.97137585771026835, 0.22193859302282001])
U = np.array([1.4808731812451179, 0.40910071361399719, 2.9182373984706249])
delta = U - L

gammaVar = np.array([1, 1, 1])  # set all decay rates as variables
edgeCounts = [1, 1, 1]  # count incoming edges to each node to structure the parameter array
parameterVar = [np.array([[np.nan for j in range(3)] for k in range(nEdge)]).squeeze() for nEdge in
                edgeCounts]  # pass all parameters as variable
f = Repressilator(gammaVar, parameterVar)

hill = 6
# p = np.concatenate([L, delta, T])
p = np.array([L[0], delta[0], T[0], L[1], delta[1], T[1], L[2], delta[2], T[2]])


def f_call(t, x):
    return f(x, hill, p)


def jac(x):
    return f.dx(x, hill, p)


y0 = [1, 1, 1]

f_call(1, y0)

# print(jac(y0))


# solutions I could find all shoot at an equilibrium
sol = solve_ivp(f_call, [0, 100], y0)

fig = plt.figure()
ax = plt.axes(projection='3d')

plt.plot(sol.y[0, :], sol.y[1, :], sol.y[2, :])
plt.plot(sol.y[0, -1], sol.y[1, -1], sol.y[2, -1], 'o')

fig2 = plt.figure()
plt.plot(sol.t, sol.y[1, :],)




print('End of the game!')
