import numpy as np
from ndma.model.model import Model

A = Model.Model_from_string("""\nX1 : (X1+X2)(~X3)\nX2 : (X1)\nX3 : (X1)(~X2)""")

print(A)

adjacency = np.array([[1, 1, 1], [1, 0, -1], [-1, 0, 0]])
B = Model.Model_from_adjacency(adjacency)
print(B)