
from numpy import array, ones


def nln4d(h=False, pl=0, mi=50):
    """Constrained 4-d nonlinear program from Hock & Schittkowski
    (test problem #71)
    """
    x0 = array([1.0, 5.0, 5.0, 1.0])
    xmin = ones(4)
    xmax = 5 * xmin


def f_71(x, user_data=None):
    """Calculates the objective value.
    """
    return x[0] * x[3] * sum(x[:3]) + x[2]


def grad_f_71(x, user_data=None):
    """Calculates gradient for objective function.
    """
    return array([ x[0] * x[3] + x[3] * sum(x[:3]),
                   x[0] * x[3],
                   x[0] * x[3] + 1,
                   x[0] * sum(x[:3]) ])


def g_71(x, user_data=None):
    """Calculates the constraint values and returns an array.
    """
    g = array( [sum(x**2) - 40.0] )
    h = array( [ -prod(x) + 25.0] )
    return r_[g, h]


def jac_g_71(x, flag, user_data=None):
    """Calculates the Jacobi matrix.

    If the flag is true, returns a tuple (row, col) to indicate the
    sparse Jacobi matrix's structure.
    If the flag is false, returns the values of the Jacobi matrix
    with length nnzj.
    """
    if flag:
        pass
    else:
        dg = sparse( 2 * x ).T
        dh = sparse( -prod(x) / x ).T
    return r_[dg, dh]


def h_71(x, lagrange, obj_factor, flag, Hl):
    """Calculates the Hessian matrix
    """
    lmbda = asscalar( lam['eqnonlin'] )
    mu    = asscalar( lam['ineqnonlin'] )
    _, _, d2f = f7(x, True)

    Lxx = sigma * d2f + lmbda * 2 * speye(4, 4) - \
        mu * sparse([
            [        0.0, x[2] * x[3], x[2] * x[3], x[1] * x[2]],
            [x[2] * x[2],         0.0, x[0] * x[3], x[0] * x[2]],
            [x[1] * x[3], x[0] * x[3],         0.0, x[0] * x[1]],
            [x[1] * x[2], x[0] * x[2], x[0] * x[1],         0.0]
        ])
    return Lxx
