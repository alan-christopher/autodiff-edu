"""
The autodiff module provides a crude and incomplete autodifferentation
implementation (only covers + and *) which is only suitable for educational
purposes.

If you're looking for an autodiff package to use for an actual project
https://pythonhosted.org/ad/ is a *much* more suitable choice.
"""

import collections
import numpy as np

##################
#  Forward Mode  #
##################


class _FDepVar(object):
    """Represents an (intermediate)? dependent variable in forward autodiff.

       If we call x the vector of independent variables, and y the dependent
       variable represented by self, then self.val == y(x) and
       self.grad[i] == y_xi(x).
    """

    def __init__(self, val, grad):
        self.val = val
        self.grad = grad

    def __add__(f, g):
        if not isinstance(g, _FDepVar):
            # (f(x) + c)' = f'
            return _FDepVar(f.val + g, f.grad)
        # (f(x) + g(x))' = f' + g'
        return _FDepVar(f.val + g.val, f.grad + g.grad)

    def __radd__(self, other):
        return self+other

    def __mul__(f, g):
        if not isinstance(g, _FDepVar):
            # (cf(x))' = cf'
            return _FDepVar(f.val * g, f.grad * g)
        # (f(x)g(x))' = f'g + fg'
        return _FDepVar(f.val * g.val,
                        f.grad * g.val + f.val * g.grad)

    def __rmul__(self, other):
        return self*other


def forward(f):
    """
    Takes a function f and returns a function f_J s.t. f_J(x)[0] is f(x) and
    f_J(x)[1] is f's Jacobian.
    """
    def f_J(*args):
        wrapped = []
        for i, arg in enumerate(args):
            # Replace the input variables with _FDepVars which track and
            # propagate both their values and their derivatives WRT the input
            # variables.
            #
            # The partial derivative of a variable WRT itself is 1. WRT another
            # independent variable is 0.
            grad = np.zeros(len(args))
            grad[i] = 1
            wrapped.append(_FDepVar(arg, grad))
        out = f(*wrapped)
        # Extract values and derivatives to hide _FDepVars from an innocent
        # world.
        try:
            return [o.val for o in out], [o.grad for o in out]
        except TypeError:
            return out.val, out.grad
    return f_J


###################
#  Backward Mode  #
###################
class _BDepVar(object):
    """Represents an (intermediate)? dependent variable in backward autodiff.

    _BDepVars store the entire graph of a calculation, weighting edges between
    nodes according to the partial derivative of one node with respect to
    another. For instance, the node (* f g) would keep edges pointing to f and
    g, with weight (g, f) respectively.

    propagate(self) can then be called from the final result of the calculation
    to calculate its derivative with respect to an input variable by
    application of the chain rule.
    """

    def __init__(self, val, edges):
        self.val = val
        self.edges = edges
        self.parents = set()
        self.derivs = collections.defaultdict(int)

    def __add__(f, g):
        if not isinstance(g, _BDepVar):
            g = _BDepVar(g, [])
        # d/df (f+g) = d/dg (f+g) = 1
        ret = _BDepVar(f.val + g.val, [(f, 1), (g, 1)])
        f.parents.add(ret)
        g.parents.add(ret)
        return ret

    def __radd__(self, other):
        return self+other

    def __mul__(f, g):
        if not isinstance(g, _BDepVar):
            g = _BDepVar(g, [])
        # d/df (fg) = g, d/dg (fg) = f
        ret = _BDepVar(f.val*g.val, [(f, g.val), (g, f.val)])
        f.parents.add(ret)
        g.parents.add(ret)
        return ret

    def __rmul__(self, other):
        return self*other

    def propagate(self, wrt, done=None):
        """
        Propagate the value of partial derivatives down the compute graph by
        way of the chain rule.
        """
        if wrt == self:
            done = set()
            # df/df = 1
            self.derivs[wrt] = 1

        # The calculation graph is guaranteed to be a DAG, but not a tree (e.g.
        # y = x*x), so we need to accumulate together every flow that passes
        # through a particular node to get the derivative of the final
        # computation with respect to that node. What's more, since this is a
        # cascading calculation, we can't propagate until all of our incoming
        # edges are addressed.
        #
        # Yes, there are more efficient ways to achieve DAG-ordering. No, I
        # don't care.
        for p in self.parents:
            if p not in done:
                return
        if self in done:
            return

        for (node, weight) in self.edges:
            node.derivs[wrt] += self.derivs[wrt] * weight
            done.add(self)
        for (node, weight) in self.edges:
            node.propagate(wrt, done=done)


def backward(f):
    """
    Takes a function f and returns a function f_J s.t. f_J(x)[0] is f(x) and
    f_J(x)[1] is f's Jacobian.
    """
    def f_J(*args):
        # Replace the input variables with _BDepVars which will remember the
        # graph of the calculation.
        wrapped = [_BDepVar(arg, []) for arg in args]
        out = f(*wrapped)
        # For each output, propagate derivative information back to the inputs
        # by way of the chain rule, then append the result to the jacobian.
        try:
            J = []
            for o in out:
                o.propagate(o)
                J.append([w.derivs[o] for w in wrapped])
            return [o.val for o in out], J
        except TypeError:
            out.propagate(out)
            return out.val, [w.derivs[out] for w in wrapped]
    return f_J
