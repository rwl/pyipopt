
from pyipopt_qp_test import qp4d, lp3d
from pyipopt_test import nln4d

def test_pyipopt():

    h = False
    pl = 0
    mi = 50

    nln4d(h, pl, mi)

    qp4d(h, pl, mi)
    lp3d(h, pl, mi)
