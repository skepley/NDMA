from ndma.activation import tanhActivation
from ndma.model import Model
import numpy as np
from matplotlib import pyplot as plt

x = np.array([4, 3, 2.], dtype=float)
gamma = np.array([1,2,3.])
p = [1,.2, 5.,1,.2, 5.,1,.2, 5.,1,.2, 5.,1,.2, 5.,1,.2, 5.]

A_hill = Model.Model_from_string("""\nX1 : (X1+X2)(~X3)\nX2 : (X1)\nX3 : (X1)(~X2)""")
p_Hill = [1,2.,3.,3.,1,2.,3.,3.,1,2.,3.,3,1,2.,3.,3.,1,2.,3.,3.,1,2.,3.,3.]
y_hill = A_hill(x, gamma, p_Hill)

y = A_hill.odeint(np.linspace(0,40.,100), x, gamma, p_Hill)
print(y, np.shape(y.y))

ax = plt.figure().add_subplot(projection='3d')
ax.plot(y.y[0,:],y.y[1,:],y.y[2,:])
plt.title('Hill model')
plt.show()


A_tanh = Model.Model_from_string("""\nX1 : (X1+X2)(~X3)\nX2 : (X1)\nX3 : (X1)(~X2)""",activationFunction=tanhActivation)
p_tanh = [1+1, 2.,3.,1+1,2.,3.,1+1,2.,3.,1+1,2.,3.,1+1,2.,3.,1+1,2.,3.]
y_tanh = A_tanh(x, gamma, p_tanh)

sol = A_tanh.odeint(np.linspace(0,40.,100), x, gamma, p_tanh)

ax = plt.figure().add_subplot(projection='3d')
ax.plot(sol.y[0,:],sol.y[1,:],sol.y[2,:])
plt.title('Tanh model')
plt.show()