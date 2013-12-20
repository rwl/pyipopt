
from numpy import array, zeros, Inf
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from scipy.sparse import csr_matrix as sparse

from pyipopt_qp import pyipopt_qp, pyipopt_lp


def lp3d(h=False, pl=0, mi=50):
    """3-d LP from linprog documentation.
    """
    c = array([-5, -4, -6], float)
    A = sparse([[1, -1,  1],
                [3,  2,  4],
                [3,  2,  0]], dtype=float)
    l = None
    u = array([20, 42, 30], float)
    xmin = array([0, 0, 0], float)
    x0 = None

    x, zl, zu, obj, status, zg = pyipopt_lp(c, A, l, u, x0, xmin, None,
                                            hessian=h,
                                            print_level=pl, max_iter=mi)

    assert status == 0  # success
    assert_array_almost_equal(x, [0, 15, 3])
    assert_almost_equal(obj, -78.0, 5)

    assert_array_almost_equal(zl, [1, 0, 0], 9)
    assert_array_almost_equal(zu, zeros(x.shape), 13)

#    assert_array_almost_equal(zg[:, 0], [0, 0, 0], 13)
#    assert_array_almost_equal(zg[:, 1], [0, 1.5, 0.5], 9)

    print '3-d LP - success'


def qp4d(h=False, pl=0, mi=50):
    """Constrained 4-d quadratic program from
    http://www.jmu.edu/docs/sasdoc/sashtml/iml/chap8/sect12.htm
    """

    H = sparse([[1003.1,  4.3,     6.3,     5.9],
                [4.3,     2.2,     2.1,     3.9],
                [6.3,     2.1,     3.5,     4.8],
                [5.9,     3.9,     4.8,    10.0]])
    c = zeros(4)
    A = sparse([[   1,       1,       1,       1],
                [0.17,    0.11,    0.10,    0.18]])
    l = array([1, 0.10])
    u = array([1, Inf])
    xmin = zeros(4)
    x0 = array([1, 0, 0, 1], float)

    x, zl, zu, obj, status, zg = pyipopt_qp(H, c, A, l, u, x0, xmin, None,
                                           hessian=h,
                                           print_level=pl, max_iter=mi)

    assert status == 0  # success
    assert_array_almost_equal(x, array([0, 2.8, 0.2, 0]) / 3)
    assert_almost_equal(obj, 3.29 / 3)

    assert_array_almost_equal(zl, [2.24, 0, 0, 1.7667], 4)
    assert_array_almost_equal(zu, zeros(x.shape), 13)

#    assert_array_almost_equal(zg[:, 0], array([6.58, 0]) / 3, 6)
#    assert_array_almost_equal(zg[:, 1], [0, 0], 13)

    print '4-d QP - success'
