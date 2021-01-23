import autodiff
import numpy.testing
import pytest


def maybe_wrap(x):
    """Convenience function to turn scalars into 1-element lists."""
    try:
        x[0]
        return x
    except TypeError:
        return [x]


def pow(x, n):
    if n == 0:
        return 1
    if n == 1:
        return x
    if n % 2 == 1:
        return x*pow(x, n-1)
    return pow(x*x, n/2)


TEST_CASES = [
    # R->R
    (lambda x: x*x, (1,), 1, 2),
    (lambda x: x*x, (7,), 49, 14),
    (lambda x: pow(x, 8), (1,), 1, 8),
    (lambda x: pow(x, 1), (1,), 1, 1),
    # R^2->R
    (lambda x, y: x*y,
     (0, 0),
     0,
     (0, 0)),
    (lambda x, y: x*y,
     (1, 2),
     2,
     (2, 1)),
    # R->R^2
    (lambda x: (sum([x]*12), pow(x, 3)),
     (3,),
     (36, 27),
     ([12], [27])),
    # R^2->R^2
    (lambda x, y: (sum([x, y]*2), x*x*y + 2*y),
     (2, 3),
     (10, 18),
     ([2, 2], [12, 6])),
]


@pytest.mark.parametrize("fun,args,f,Jf", TEST_CASES)
def test_forward_autodiff(fun, args, f, Jf):
    f_emp, Jf_emp = autodiff.forward(fun)(*args)
    f_emp = maybe_wrap(f_emp)
    f = maybe_wrap(f)
    Jf_emp = maybe_wrap(Jf_emp)
    Jf = maybe_wrap(Jf)
    numpy.testing.assert_array_equal(f_emp, f)
    numpy.testing.assert_array_equal(Jf_emp, Jf)


@pytest.mark.parametrize("fun,args,f,Jf", TEST_CASES)
def test_backward_autodiff(fun, args, f, Jf):
    f_emp, Jf_emp = autodiff.backward(fun)(*args)
    f_emp = maybe_wrap(f_emp)
    f = maybe_wrap(f)
    Jf_emp = maybe_wrap(Jf_emp)
    Jf = maybe_wrap(Jf)
    numpy.testing.assert_array_equal(f_emp, f)
    numpy.testing.assert_array_equal(Jf_emp, Jf)
