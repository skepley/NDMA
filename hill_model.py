"""
Classes and methods for constructing, evaluating, and doing parameter continuation of Hill Models

    Author: Shane Kepley
    Email: s.kepley@vu.nl
    Created: 2/29/2020
"""
import numpy as np
from scipy import optimize

# ignore overflow and division by zero warnings:
from coordinate.coordinate import Coordinate
from model.model import Model

np.seterr(over='ignore', invalid='ignore')


def npA(size, dim=2):
    """Return a random square integer matrix of given size for testing numpy functions."""
    A = np.random.randint(1, 10, dim * [size])
    return np.asarray(A, dtype=float)


def is_vector(array):
    """Returns true if input is a numpy vector i.e. has shape (n,). Returns false for column or row vectors i.e.
    objects with shape (n,1) or (1,n)."""

    return len(np.shape(array)) == 1


def ezcat(*coordinates):
    """A multiple dispatch concatenation function for numpy arrays. Accepts arbitrary inputs as int, float, tuple,
    list, or numpy array and concatenates into a vector returned as a numpy array. This is recursive so probably not
    very efficient for large scale use."""

    if len(coordinates) == 1:
        if isinstance(coordinates[0], list):
            return np.array(coordinates[0])
        elif isinstance(coordinates[0], np.ndarray):
            return coordinates[0]
        else:
            return np.array([coordinates[0]])

    try:
        return np.concatenate([coordinates[0], ezcat(*coordinates[1:])])
    except ValueError:
        return np.concatenate([np.array([coordinates[0]]), ezcat(*coordinates[1:])])


def find_root(f, Df, initialGuess, diagnose=False):
    """Default root finding method to use if one is not specified"""

    solution = optimize.root(f, initialGuess, jac=Df, method='hybr')  # set root finding algorithm
    if diagnose:
        return solution  # return the entire solution object including iterations and diagnostics
    else:
        return solution.x  # return only the solution vector


def full_newton(f, Df, x0, maxDefect=1e-13):
    """A full Newton based root finding algorithm"""

    def is_singular(matrix, rank):
        """Returns true if the derivative becomes singular for any reason"""
        return np.isnan(matrix).any() or np.isinf(matrix).any() or np.linalg.matrix_rank(matrix) < rank

    fDim = len(x0)  # dimension of the domain/image of f
    maxIterate = 100

    if not is_vector(x0):  # an array whose columns are initial guesses
        print('not implemented yet')

    else:  # x0 is a single initial guess
        # initialize iteration
        x = x0.copy()
        y = f(x)
        Dy = Df(x)
        iDefect = np.linalg.norm(y)  # initialize defect
        iIterate = 1
        while iDefect > maxDefect and iIterate < maxIterate and not is_singular(Dy, fDim):
            if fDim == 1:
                x -= y / Dy
            else:
                x -= np.linalg.solve(Dy, y)  # update x

            y = f(x)  # update f(x)
            Dy = Df(x)  # update Df(x)
            iDefect = np.linalg.norm(y)  # initialize defect
            iIterate += 1

        if iDefect < maxDefect:
            return x
        else:
            print('Newton failed to converge')
            return np.nan


def verify_call(func):
    """Evaluation method decorator for validating evaluation calls. This can be used when troubleshooting and
    testing and then omitted in production runs to improve efficiency. This decorates any HillModel or
    HillCoordinate method which has inputs of the form (self, x, *parameter, **kwargs)."""

    def func_wrapper(*args, **kwargs):
        hillObj = args[0]  # a HillModel or HillCoordinate instance is passed as 1st position argument to evaluation
        # method
        x = args[1]  # state vector passed as 2nd positional argument to evaluation method
        if not is_vector(x):  # x must be a vector
            raise TypeError('Second argument must be a state vector.')

        if issubclass(type(hillObj), Coordinate):
            N = hillObj.nState
            parameter = args[2]  # parameter vector passed as 3rd positional argument to evaluation method
        elif issubclass(type(hillObj), Model):
            N = hillObj.dimension
            parameter = hillObj.parse_parameter(*args[2:])  # parameters passed as variable argument to
            # evaluation method and need to be parsed to obtain a single parameter vector
        else:
            raise TypeError('First argument must be a HillCoordinate or HillModel instance. Instead it received {'
                            '0}'.format(type(hillObj)))

        if len(x) != N:  # make sure state input is the correct size
            raise IndexError(
                'State vector for this evaluation should be size {0} but received a vector of size {1}'.format(N,
                                                                                                               len(x)))
        elif len(parameter) != hillObj.nParameter:
            raise IndexError(
                'Parsed parameter vector for this evaluation should be size {0} but received a vector of '
                'size {1}'.format(
                    hillObj.nParameter, len(parameter)))

        else:  # parameter and state vectors are correct size. Pass through to evaluation method
            return func(*args, **kwargs)

    return func_wrapper


PARAMETER_NAMES = ['ell', 'delta', 'theta', 'hillCoefficient']  # ordered list of HillComponent parameter names


