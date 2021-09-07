from hill_model import *
from saddle_finding_functionalities import *
from toggle_switch_heat_functionalities import *
import matplotlib.pyplot as plt
import sys
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
from models.TS_model import ToggleSwitch

# define the saddle node problem for the toggle switch
decay = np.array([1, np.nan], dtype=float)
p1 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_2, delta_2, theta_2)
f = ToggleSwitch(decay, [p1, p2])
SN = SaddleNode(f)

# size of the sample
n_sample_side = 5
n_sample = (n_sample_side)**2
n_second_sample = 5
# a random parameter list
interpolation_array = np.array([np.linspace(0.95, 1.05, n_sample_side)])
[u, v] = np.meshgrid(np.linspace(0.85, 1.1, n_sample_side), np.linspace(1.1, 1.9, n_sample_side))
u = u.flatten()
u = np.repeat(u, n_second_sample)
v = v.flatten()
v = np.repeat(v, n_second_sample)
a = np.array([fiber_sampler(u[j], v[j]) for j in range(n_sample*n_second_sample)])

n_sample = n_sample * n_second_sample

parameter_full = np.empty(shape=[0, 5])
solutions = np.empty(0)
bad_parameters = np.empty(shape=[0, 5])
bad_candidates = []
boring_parameters = np.empty(shape=[0, 5])
multiple_saddles = np.empty(shape=[0, 5])
for j in []:#range(n_sample):#range(n_sample):
    a_j = a[j, :]
    SNParameters, badCandidates = find_saddle_coef(f, [1,2,3,4,5,10,20,30,40,50], a_j)
    if SNParameters and SNParameters is not 0:
        for k in range(len(SNParameters)):
            #print('Saddle detected')
            parameter_full = np.append(parameter_full, [a_j], axis=0)
            solutions = np.append(solutions, SNParameters[k])
            if k > 0:
                c = parameter_to_DSGRN_coord(a_j, 10)
                print('More than one saddle detected at coord = ', c)
                multiple_saddles = np.append(multiple_saddles, [a_j], axis=0)
    if badCandidates and badCandidates is not 0:
        print('\nA bad parameter')
        bad_parameters = np.append(bad_parameters, [a_j], axis=0)
        bad_candidates.append(badCandidates)
    printing_statement = 'Completion: ' + str(j) + ' out of ' + str(n_sample)
    sys.stdout.write('\r' + printing_statement)
    sys.stdout.flush()

    if SNParameters is 0 and badCandidates is 0:
        #print('boooring')
        boring_parameters = np.append(boring_parameters, [a_j], axis=0)

#if np.size(multiple_saddles, 1) > 1:
    #np.savez('looking_for_isolas',
    #        u=u, v=v, a=a, parameter_full=parameter_full, solutions=solutions, multiple_saddles=multiple_saddles)
# stopHere

data = np.load('boundary_averaging_data.npz', allow_pickle=True)

#multiple_saddles = data.f.multiple_saddles
parameter_full = data.f.parameter_full
solutions = data.f.solutions

# ISOLAS CAN BE PLOTTED HERE
"""for j in multiple_saddles:
    a_j = j
    SNParameters, badCandidates = find_saddle_coef(f, [1,2,3,4,5,10,20,30,40,50], a_j)
    if len(SNParameters) <= 1:
        print('that is not what we were expecting')
    if len(SNParameters) > 2:
        print('that is also quite unexpected')
    interesting_hills = np.linspace(SNParameters[0][0]*1.3, SNParameters[1][0]*0.8, 300)
    for hill in interesting_hills:
        equilibria = HillModel.find_equilibria(f, 5, hill, a_j)
        plt.plot(equilibria[:, 0], np.repeat(hill, np.size(equilibria, 0)),'.')
        """

# what happens to the other points outside the center region?

c = parameter_to_DSGRN_coord(parameter_full, 10)
out_of_center = [np.linalg.norm([c[0][j]-1.5, c[1][j]-1.5], np.inf) for j in range(np.size(c, 1))]
out_parameters = [parameter_full[j, :] for j in range(len(out_of_center)) if out_of_center[j]>0.5]
"""
-------------------------
theta1 = 1/2
L1 = .26
U1 = .49

theta2 = 1/2
L2 =.05
U2 = .55
-------------------------
theta1 = 1/2
L1 = .248
U1 = .48

theta2 = 1/2
L2 =.1077
U2 = .4923
-------------------------
theta1 = 1/2
L1 = .05
U1 = .55

theta2 = 1/2
L2 =.26
U2 = .49
-------------------------
"""
# out_parameters = [np.array([1, 1/2,.26,.49-.26,1,0.5,.05,.55-.05])]
# out_parameters = [np.array([1,1/2,.248,.48-.248,1,1/2,0.1077,.4923-0.1077])]
# out_parameters = [np.array([1,1/1,.05,.55-.05,1,1/2,.26,.49-.26])]
counter = 1
for j in out_parameters:
    a_j = j
    if counter >=0:#== 2 or counter == 5 or counter == 29:
        SNParameters, badCandidates = find_saddle_coef(f, [1, 10, 50, 100, 200], a_j)
        if len(SNParameters) > 1:
            print('that is not what we were expecting')
        if len(SNParameters) < 1:
            print('that is also quite unexpected')
        if len(SNParameters) != 2:
            print('ONLY '+str(len(SNParameters))+' SADDLES FOUND\n' +str(counter) + ' p=' + str(a_j))
            continue
        print(SNParameters)
        interesting_hills = np.linspace(SNParameters[0][0]*1.3, SNParameters[1][0]*0.8, 100)
        plt.figure()
        for hill in interesting_hills:
            equilibria = HillModel.find_equilibria(f, 5, hill, a_j)
            plt.plot(equilibria[:, 0], np.repeat(hill, np.size(equilibria, 0)), 'b.')
        x, y = parameter_to_DSGRN_coord(np.array([a_j]), 10)
        print(str(counter)+' p='+str(a_j)+', (x,y)'+str(x)+str(y))
        plt.savefig('figure'+str(counter)+'.png')
    counter = counter+1

stopHere

a_j = out_parameters[0]
SNParameters, badCandidates = find_saddle_coef(f, [1, 10, 50, 100, 300], a_j)
interesting_hills = np.sort(np.linspace(SNParameters[1][0]*0.8, SNParameters[0][0]*1.3, 100))
x_data = []
y_data = []

fig, ax = plt.subplots()
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
line1, line2, = ax.plot(0, 0, 'g-', 1, 1, 'r-')


def animation_frame(hill):
    # hill = interesting_hills[i]
    """x_data = np.arange(-5, 5, 0.1)
    y_data = np.sin(i*x_data)

    line1.set_xdata(x_data)
    line1.set_ydata(y_data)
    line2.set_xdata([0, 3])
    line2.set_ydata([9, 0])"""
    x1, y1, x2, y2 = f.plot_nullcline(hill, a_j, domainBounds=((0, 2), (0, 2)), nNodes=200)

    line1.set_xdata(x1)
    line1.set_ydata(y1)
    line2.set_xdata(x2)
    line2.set_ydata(y2)

    return line1, line2,


anim = FuncAnimation(fig, func=animation_frame, frames=interesting_hills, interval=400)
# plt.show()

writervideo = FFMpegWriter(fps=60)
anim.save('nullclines_in_isolas.mp4', writer=writervideo)
plt.close()

print('It is the end!')
