
"""IPOPT based quadratic program solver.
"""

import pyipopt

from numpy import zeros, Inf, ones, dot

from scipy.sparse import tril, csc_matrix as sparse


def pyipopt_qp(H, c, A, gl, gu, x0=None, xl=None, xu=None,
               hessian=False, **kw_args):
    """IPOPT based quadratic program solver.

    Solves the quadratic programming problem:

          min 0.5*x'Hx + c'x
           x

    subject to:
            gl <= Ax <= gu
            xl <=  x <= xu

    @param H: quadratic cost coefficient matrix
    @param c: linear cost coefficients
    @param A: linear constraint matrix
    @param gl: lower bound of constraints
    @param gu: upper bound of constraints
    @param x0: initial starting point
    @param xl: lower bound of x as bounded constraints
    @param xu: upper bound of x as bounded constraints
    @param finite_differences: evaluate Hessian by finite differences
    @param kw_args: IPOPT options
    """
    userdata = {'H': H, 'c': c, 'A': A}

    if x0 is None:
        x0 = zeros(A.shape[1])

    # n is the number of variables
    n = x0.shape[0]
    # number of linear constraints (zero nln)
    m = A.shape[0]
    # number of nonzeros in Jacobi matrix
    nnzj = A.nnz

    if gl is None:
        gl = -Inf * ones(m)
    if gu is None:
        gu =  Inf * ones(m)

    if (xl is None) or (len(xl) == 0):
        xl = -Inf * ones(n)
    if (xu is None) or (len(xu) == 0):
        xu =  Inf * ones(n)

    if not hessian:
        nnzh = 0
        nlp = pyipopt.create(n, xl, xu, m, gl, gu, nnzj, nnzh,
                             qp_f, qp_grad_f, qp_g, qp_jac_g)
    else:
        Hl = tril(H, format='coo')

        # number of non-zeros in Hessian matrix
        nnzh = Hl.nnz

        eval_h = lambda x, lagrange, obj_factor, flag, \
                userdata=None: qp_h(x, lagrange, obj_factor, flag, Hl)

        nlp = pyipopt.create(n, xl, xu, m, gl, gu, nnzj, nnzh,
                             qp_f, qp_grad_f, qp_g, qp_jac_g, eval_h)

    for k, v in kw_args.iteritems():
        if isinstance(v, int):
            nlp.int_option(k, v)
        elif isinstance(v, basestring):
            nlp.str_option(k, v)
        else:
            nlp.num_option(k, v)


    # returns  x, upper and lower bound for multiplier, final
    # objective function obj and the return status of IPOPT
    result = nlp.solve(x0, m, userdata)
    ## final values for the primal variables
    x = result[0]
    ## final values for the lower bound multipliers
    zl = result[1]
    ## final values for the upper bound multipliers
    zu = result[2]
    ## final value of the objective
    obj = result[3]
    ## status of the algorithm
    status = result[4]
    ## final values for the constraint multipliers
    zg = result[5]

    nlp.close()

    return x, zl, zu, obj, status, zg


def pyipopt_lp(c, A, gl, gu, x0, xl=None, xu=None, hessian=False, **kw_args):
    """IPOPT based quadratic program solver.

    Solves linear programs of the form:

        min c'x
         x

    subject to:

            gl <= Ax <= gu
            xl <=  x <= xu
    """
    n = c.shape[0]
    H = sparse((n, n))

    return pyipopt_qp(H, c, A, gl, gu, x0, xl, xu, hessian, **kw_args)


def qp_f(x, user_data=None):
    """Calculates the objective value.
    """
    return 0.5 * dot(x * user_data['H'], x) + dot(user_data['c'], x)


def qp_grad_f(x, user_data=None):
    """Calculates gradient for objective function.
    """
    return user_data['H'] * x + user_data['c']


def qp_g(x, user_data=None):
    """Calculates the constraint values and returns an array.
    """
    return user_data['A'] * x


def qp_jac_g(x, flag, user_data=None):
    """Calculates the Jacobi matrix.

    If the flag is true, returns a tuple (row, col) to indicate the
    sparse Jacobi matrix's structure.
    If the flag is false, returns the values of the Jacobi matrix
    with length nnzj.
    """
    Acoo = user_data['A'].tocoo()
    if flag:
        return (Acoo.row, Acoo.col)
    else:
        return Acoo.data


#def qp_h(x, lagrange, obj_factor, flag, user_data=None):
#    """Calculates the Hessian matrix
#    """
#    H = tril(user_data['H'], format='coo')
#    if flag:
#        return (H.row, H.col)
#    else:
#        return H.data


def qp_h(x, lagrange, obj_factor, flag, Hl):
    """Calculates the Hessian matrix
    """
    if flag:
        return (Hl.row, Hl.col)
    else:
        return Hl.data
