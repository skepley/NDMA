import numpy as np
from ndma.model.model import Model
from ndma.activation.tanhActivation import  tanhActivation
from matplotlib import pyplot as plt
def test_tanhActivation():
    H = tanhActivation(1, ell=3.4, delta=1., theta=2.3)
    x1 = 2.
    print('tanh =', H(x1))

    A = Model.Model_from_string("""\nX1 : (X1+X2)(~X3)\nX2 : (X1)\nX3 : (X1)(~X2)""",activationFunction=tanhActivation)

    print(A)

    x = np.array([4, 3, 2.], dtype=float)

    # check f evaluation
    # print('f eval = ')
    gamma = np.array([1,2,3.])
    p = [1,.2, 5.,1,.2, 5.,1,.2, 5.,1,.2, 5.,1,.2, 5.,1,.2, 5.]
    y = A(x, gamma, p)
    # print(y)
    assert True
