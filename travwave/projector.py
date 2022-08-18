from __future__ import division

import numpy as np
from . import newton


class Hamiltonian_projector(object):
    def __init__(self, equation):
        self.equation = equation
        
    def construct(self, u, mu):
        """
        Attaches the solution wave and extent parameter together for further computation.
        """
        return np.hstack([ np.hstack(u), mu ])

    def destruct(self, vector):
        """
        Separates the solution wave and the extent parameter.
        """
        N = self.equation.Ngrid
        u = np.array([ vector[ : N], vector[N : N+N] ]) 
        mu = vector[-1]
        return u, mu

    def project(self, guess_wave, H0):
        """
        Runs a Newton solver on a system of nonlinear equations once. Takes the residual(vector) as the system to solve.
        """
        def residual(vector):
            """
            Contructs a system of nonlinear equations.
            """
            wave, mu = self.destruct(vector)
            
            res = wave - mu * self.grad_Hamiltonian(wave) - guess_wave
            return self.construct( res,  self.equation.Hamiltonian(wave) - H0)

        guess = self.construct(guess_wave, mu = 0)
        nsolver = newton.MultipleSolver(residual)
        computed = nsolver.run(guess)
        wave, mu = self.destruct(computed)
        return wave

    def Hamiltonian(self, y):
        N = self.equation.Ngrid
        res = np.array([ y[ : N], y[N : ] ])
        return self.equation.Hamiltonian(res)

    def grad_Hamiltonian(self, wave):
        N = self.equation.Ngrid
        res = newton.jacobian( self.Hamiltonian, np.hstack([ wave[0], wave[1] ]) )
        #-------------------------
        #Alternative gradient:
        #res = 0
        #for i in range(self.equation.num_unknowns):
        #    for k in range(self.equation.num_right_terms[i]):
        #        res = res + self.equation[i, k](wave)
        #-------------------------
        return np.array([ res[ : N], res[N : ] ])       