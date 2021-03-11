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


""" These parameters are good for either variation of the repressilator
L[x1->x2]" : 0.9132235573212335, "L[x2->x3]" :
0.08856225631514987, "L[x3->x1]" : 0.081866957498757928, "T[x1->x2]" :
0.14056968648960766, "T[x2->x3]" : 0.97137585771026835, "T[x3->x1]" :
0.22193859302282001, "U[x1->x2]" : 1.4808731812451179, "U[x2->x3]" :
0.40910071361399719, "U[x3->x1]" : 2.9182373984706249 """


L = np.array([0.9132235573212335, 0.08856225631514987, 0.081866957498757928])
T = np.array([0.14056968648960766, 0.97137585771026835, 0.22193859302282001])
U = np.array([1.4808731812451179, 0.40910071361399719, 2.9182373984706249])
delta = U - L

gammaVar = np.array([1, 1, 1])  # set all decay rates as variables
edgeCounts = [1, 1, 1]  # count incoming edges to each node to structure the parameter array
parameterVar = [np.array([[np.nan for j in range(3)] for k in range(nEdge)]).squeeze() for nEdge in
                edgeCounts]  # pass all parameters as variable
f = Repressilator(gammaVar, parameterVar)

hill = 10
# p = np.concatenate([L, delta, T])
p = np.array([L[0], delta[0], T[0], L[1], delta[1], T[1], L[2], delta[2], T[2]])


def f_call(t, x):
    return f(x, hill, p)


def jac(x):
    return f.dx(x, hill, p)


y0 = np.random.random(size = 3)

f_call(1, y0)



"""
# pretty pictures showing what a Hill function looks like

f = lambda x, l, delta, theta, n : l + delta * theta**n /(x**n + theta**n)
x = np.linspace(0, 15, 1500)
f_x = [f(x_loc, 1, 2, 4, 4) for x_loc in x]
plt.plot(x, f_x)
fig = plt.figure()
for theta_loc in [3,4,5,6,7]:
    f_x = [f(x_loc, 1, 2, theta_loc, 4) for x_loc in x]
    plt.plot(x, f_x)
fig = plt.figure()

for n_loc in [3,4,6,9,13,15]:
    f_x = [f(x_loc, 1, 2, 4, n_loc) for x_loc in x]
    plt.plot(x, f_x)
fig = plt.figure()
for n_loc in [3, 4, 5, 6, 7]:
    theta_loc = n_loc
    f_x = [f(x_loc, 1, 2, theta_loc, n_loc) for x_loc in x]
    plt.plot(x, f_x)

"""
f.dx(y0, hill, p)


for i in range(10):
    y0 = np.random.random(size = 3)
    # solutions I could find all shoot at an equilibrium
    sol = solve_ivp(f_call, [0, 100], y0)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    plt.plot(sol.y[0, :], sol.y[1, :], sol.y[2, :])
    plt.plot(sol.y[0, -1], sol.y[1, -1], sol.y[2, -1], 'o')

    fig2 = plt.figure()
    plt.plot(sol.t, sol.y[1, :])





print('End of the game!')
