from __future__ import division
import numpy as np
from travwave.equations import *
from numpy import linalg


class Petviashvili(object):
    def __init__(
        self, equation, Froude, tolerance = 10**(-12), Nmax_iter = 1000
    ):
        self.equation = equation
        self.tolerance = tolerance
        self.Nmax_iter = Nmax_iter
        self.Froude = Froude
        self.inverse_detL = 1 / ( Froude**2 - equation.thHD_over_HD )
        if type(equation)==bidirectional_whitham.Kalisch:
            self.N = self.Hur_N
            self.L = self.Kalisch_L
            self.invL = self.Kalisch_invL
        elif type(equation)==bidirectional_whitham.Hur:
            self.N = self.Hur_N
            self.L = self.Hamiltonian_Hur_L
            self.invL = self.Hamiltonian_Hur_invL
        elif type(equation)==bidirectional_whitham.Hamiltonian_Hur:
            self.N = self.Hamiltonian_Hur_N
            self.L = self.Hamiltonian_Hur_L
            self.invL = self.Hamiltonian_Hur_invL
        elif type(equation)==bidirectional_whitham.Matsuno:
            raise NotImplementedError()
        elif type(equation)==bidirectional_whitham.RS:
            raise NotImplementedError()
        else:
            raise NotImplementedError()
      
        
    def iterate(self, u0):  #u0 is an initial guess
        u = u0
        error = np.inf
        iter_num = 0
        while error > self.tolerance:
            v = self.iteration(u)
            error = linalg.norm(v - u, np.inf)
            u = v
            iter_num += 1
            if iter_num > self.Nmax_iter:
                raise NotImplementedError()
        return u
        
    def iteration(self, u):
        return self.S(u)**2 * self.invL( self.N(u) )
        
    def apply_inverse_detL(self, u):
        return self.equation.apply_operator(u, self.inverse_detL).real
        
    def S(self, u):
        Lu = self.L(u)
        resL = self.equation.integrate( Lu[0] * u[0] + Lu[1] * u[1] )
        Nu = self.N(u)
        resN = self.equation.integrate( Nu[0] * u[0] + Nu[1] * u[1] )
        return resL / resN

    def Hamiltonian_Hur_N(self, u):
        eta = u[0]
        u = u[1]
        res1 = self.equation.apply_thHD_over_HD( eta*u )
        res2 = self.equation.apply_thHD_over_HD( 0.5 * u**2 )
        return np.array([ res1, res2 ])
        
    def Hamiltonian_Hur_L(self, u):
        eta = u[0]
        u = u[1]
        F = self.Froude
        res1 = F * eta - u
        res2 = F * u - self.equation.apply_thHD_over_HD( eta )
        return np.array([ res1, res2 ])
        
    def Hamiltonian_Hur_invL(self, u):
        eta = u[0]
        u = u[1]
        F = self.Froude
        res1 = F * eta + u
        res2 = F * u + self.equation.apply_thHD_over_HD( eta )
        res1 = self.apply_inverse_detL(res1)
        res2 = self.apply_inverse_detL(res2)
        return np.array([ res1, res2 ])

    def Hur_N(self, u):
        eta = u[0]
        u = u[1]
        return np.array([ eta*u, 0.5 * u**2 ])
        
    def Kalisch_L(self, u):
        eta = u[0]
        u = u[1]
        F = self.Froude
        res1 = F * eta - self.equation.apply_thHD_over_HD( u )
        res2 = F * u - eta
        return np.array([ res1, res2 ])
        
    def Kalisch_invL(self, u):
        eta = u[0]
        u = u[1]
        F = self.Froude
        res1 = F * eta + self.equation.apply_thHD_over_HD( u )
        res2 = F * u + eta
        res1 = self.apply_inverse_detL(res1)
        res2 = self.apply_inverse_detL(res2)
        return np.array([ res1, res2 ])
        
