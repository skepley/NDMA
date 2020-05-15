"""
One line description of what the script performs (H1 line)
  
Optional file header info (to give more details about the function than in the H1 line)
Optional file header info (to give more details about the function than in the H1 line)
Optional file header info (to give more details about the function than in the H1 line)

    Output: output
    Other files required: none
    See also: OTHER_SCRIPT_NAME,  OTHER_FUNCTION_NAME
   
    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 4/24/20; Last revision: 4/24/20
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from hill_model import ToggleSwitch, SaddleNode, find_root

def SN_call_temp(SNinstance, parameter, u0):
    """Temporary SaddleNode call outside the main class definition"""

    return find_root(lambda u: SNinstance.zero_map(u, parameter), lambda u: SNinstance.diff_zero_map(u, parameter), u0,
                     diagnose=True)


# set some parameters to test using MATLAB toggle switch for ground truth
decay = np.array([np.nan, np.nan], dtype=float)  # gamma
p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)

f = ToggleSwitch(decay, [p1, p2])
f1 = f.coordinates[0]
f2 = f.coordinates[1]
H1 = f1.components[0]
H2 = f2.components[0]
n0 = 4.1

x0 = np.array([4, 3])
p0 = np.array([1, 1, 5, 3, 1, 1, 6, 3],
              dtype=float)  # (gamma_1, ell_1, delta_1, theta_1, gamma_2, ell_2, delta_2, theta_2)
# print(f(x0, n0, p0))
SN = SaddleNode(f)

# ==== find saddle node for a parameter choice
p0 = np.array([1, 1, 5, 3, 1, 1, 6, 3], dtype=float)
v0 = np.array([1, -.7])
eq0 = f.find_equilibria(10, n0, p0)
x0 = eq0[:, -1]
u0 = np.concatenate((x0, v0, np.array(n0)), axis=None)
u0Sol = SN_call_temp(SN, p0, u0)
# print(u0Sol)
x0Sol, v0Sol, n0Sol = [u0Sol.x[idx] for idx in [[0, 1], [2, 3], [4]]]
# # compare to u0Sol = [ 4.55637172,  2.25827744,  0.82199933, -0.56948846,  3.17447061]
#
# # plot nullclines and equilibria
# plt.close('all')
# plt.figure()
# f.plot_nullcline(n0, p0)
# plt.title('p = {0}; n = {1}'.format(p0, u0[-1]))
# plt.figure()
# f.plot_nullcline(n0Sol, p0)
# plt.title('p = {0}; n = {1}'.format(p0, n0Sol[0]))


# ==== This one finds a pitchfork bifurcation instead
p1 = np.array([1, 1, 5, 3, 1, 1, 5, 3], dtype=float)

v1 = np.array([1, -.7])
eq1 = f.find_equilibria(10, n0, p1)
x1 = eq1[:, -1]
u1 = np.concatenate((x1, v1, np.array(n0)), axis=None)
u1Sol = SN_call_temp(SN, p1, u1)
# print(u1Sol)
x1Sol, v1Sol, n1Sol = [u1Sol.x[idx] for idx in [[0, 1], [2, 3], [4]]]

# # plot nullclines
# plt.figure()
# f.plot_nullcline(n0, p1)
# plt.title('p = {0}; n = {1}'.format(p1, u0[-1]))
# plt.figure()
# f.plot_nullcline(n1Sol, p1)
# plt.title('p = {0}; n = {1}'.format(p1, n1Sol[0]))

# ==== Animation of continuation in N
# create a an array of Hill coefficients to plot
nFrame = 25
nRange = np.linspace(5, n1Sol, nFrame)
nNodes = 100
domainBounds = (10, 10)
Xp = np.linspace(0, domainBounds[0], nNodes)
Yp = np.linspace(0, domainBounds[1], nNodes)

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, domainBounds[0]), ylim=(0, domainBounds[1]))
ax.set_aspect('equal')

N1, = ax.plot([], [], lw=2)
N2, = ax.plot([], [], lw=2)
eq, = ax.plot([], [], 'o')
time_template = 'N = %.4f'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def snapshot_data(hillModel, N, parameter):
    """Get nullcline and equilibria data at a value of N and parameter"""

    equi = hillModel.find_equilibria(10, N, parameter)
    Z = np.zeros_like(Xp)

    # unpack decay parameters separately
    gamma = np.array(list(map(lambda f_i, parm: f_i.curry_gamma(parm)[0], hillModel.coordinates,
                              hillModel.unpack_variable_parameters(hillModel.parse_parameter(N, parameter)))))
    null1 = (hillModel(np.row_stack([Z, Yp]), N, parameter) / gamma[0])[0, :]  # f1 = 0 nullcline
    null2 = (hillModel(np.row_stack([Xp, Z]), N, parameter) / gamma[1])[1, :]  # f2 = 0 nullcline

    return null1, null2, equi


def init():
    N1.set_data([], [])
    N2.set_data([], [])
    eq.set_data([], [])
    time_text.set_text('')
    return N1, N2, eq, time_text


parm = p1


def animate(i):
    null1, null2, equilibria = snapshot_data(SN.model, nRange[i], parm)

    N1.set_data(Xp, null2)
    N2.set_data(null1, Yp)
    if equilibria.ndim == 0:
        pass
    elif equilibria.ndim == 1:
        eq.set_data([equilibria[0], equilibria[1]])
    else:
        eq.set_data(equilibria[0, :], equilibria[1, :])
    time_text.set_text(time_template % nRange[i])

    return N1, N2, eq, time_text


ani = animation.FuncAnimation(fig, animate, range(1, len(nRange)),
                              interval=10, blit=True, init_func=init, repeat_delay=1000)
plt.show()
