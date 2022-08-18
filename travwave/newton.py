# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import numpy.linalg as nl

import logging

def jacobian(F,x,h=1e-6):
    """
    Numerical Jacobian at x.
    """
    x = np.array(x)
    L = x.size
    vhs = h * np.identity(L)
    Fs = np.array([F(x+vh) for vh in vhs])
    grad = (Fs - F(x))/h
    return np.array(grad).T

class RootSolver(object):
    def __init__(self, F=None, level=0.):
        self.F = F
        self.level = level

    def residual(self, x):
        a = x.reshape(self.shape)
        res = self.F(a)
        try:
            res_vec = res.ravel()
        except AttributeError: # a list of arrays
            res_vec = np.hstack([comp.ravel() for comp in res])
        return res_vec

    def get_initial(self, x0):
        if np.isscalar(x0):
            x = np.array([x0])
        else:
            x = np.array(x0)
        self.shape = x.shape
        x = x.ravel()
        return x

    def get_result(self, x):
        return x.reshape(self.shape)

    class DidNotConverge(Exception):
        """
        Exception raised when the non linear solver does not converge.
        """

class Newton(RootSolver):
    """
    Simple Newton solver to solve F(x) = level.
    """

    h = 1e-6
    def der(self, x):
        return jacobian(self.residual, x, self.h)


    maxiter = 600
    tol = 1e-11
    def run(self, x0):
        """
        Run the Newton iteration.
        """
        x = self.get_initial(x0)
        for i in range(self.maxiter):
            d = self.der(x)
            y = self.level - self.residual(x)
            if np.isscalar(y):
                incr = y/d.item()
            else:
                try:
                    incr = np.linalg.solve(d, y)
                except np.linalg.LinAlgError as ex:
                    eigvals, eigs = np.linalg.eig(d)
                    zerovecs = eigs[:, np.abs(eigvals) < 1e-10]
                    raise np.linalg.LinAlgError("%s: %s %s" % (ex.message, repr(zerovecs), eigvals))
##                  raise np.linalg.LinAlgError("Condition: %.1e" % np.linalg.cond(d))
            x += incr
            if nl.norm(incr) < self.tol:
                break
        else:
            raise self.DidNotConverge(u"Newton algorithm did not converge after %d iterations. Dx=%.2e" % (i, nl.norm(incr)))
        self.required_iter = i
        return self.get_result(x)

    def is_zero(self, x): # use np.allclose?
        res = nl.norm(self.F(x) - self.level)
        return res < self.tol

class FSolve(RootSolver):
    """
    Wrapper around scipy.optimize.fsolve
    """
    xtol = 1.49012e-08 # default fsolve xtol
    def run(self, x0):
        guess = self.get_initial(x0)
        import scipy.optimize
        full_output = scipy.optimize.fsolve(self.residual, guess, full_output=True,  xtol = self.xtol, )
        result, self.info, success, msg = full_output
        if success != 1:
            raise self.DidNotConverge(msg)
        return self.get_result(result)

class MultipleSolver(object):
    solver_classes = [FSolve, Newton]
    def __init__(self, residual=None, level=0):
        self.solvers = [S(residual,level) for S in self.solver_classes]

    def run(self, guess):
        for solver in self.solvers:
            try:
                root = solver.run(guess)
            except solver.DidNotConverge as e:
                last_exception = e
                logging.info("Switch nonlinear solver")
            else:
                return root
        else:
            raise last_exception
