from __future__ import division
#import scipy.fftpack
#import scipy.interpolate
import numpy as np

from .base import Equation


class Whitham(Equation):
    def __init__(
        self, length, Ngrid, gravity, water_depth
    ):
        num_right_terms = np.array([2, 2])
        super(Whitham, self).__init__(length, Ngrid, 2, num_right_terms)
        self.gravity = gravity
        self.depth = water_depth
        ###
        D = self.frequencies
        thHD = np.tanh( water_depth * D)
        self.U = np.sqrt(gravity * D * thHD)
        self.thHD_over_HD = np.hstack([ 1., thHD[1:] / D[1:] / water_depth ])
        self.HD_over_thHD = 1. / self.thHD_over_HD
        self.sqrt_gD_over_thHD = np.sqrt(gravity/water_depth) * self.thHD_over_HD**(-0.5)
        self.initialize_linear_symbols()

        
    def __getitem__(self, index):
        if index == (0, 0):
            res = self.F00
        elif index == (0, 1):
            res = self.F01
        elif index == (1, 0):
            res = self.F10
        elif index == (1, 1):
            res = self.F11
        else:
            raise NotImplementedError()
        return res

    def F00(self, u, t = 0):
        u = u[1]
        res = self.apply_operator(u, self.symbol00)
        res = res.real
        return np.array([ res, np.zeros_like(res) ])
        
    def F01(self, u, t = 0):
        eta = u[0]
        u = u[1]
        res = self.apply_operator(eta*u, self.symbol01)
        res = res.real
        return np.array([ res, np.zeros_like(res) ])
        
    def F10(self, u, t = 0):
        eta = u[0]
        res = self.apply_operator(eta, self.symbol10)
        res = res.real
        return np.array([ np.zeros_like(res), res ])
        
    def F11(self, u, t = 0):
        u = u[1]
        res = self.apply_operator(u**2, self.symbol11)
        res = res.real
        return np.array([ np.zeros_like(res), res ])

    def linear_symbol00(self, t):
        U = self.U
        return np.cos(U*t)

    def linear_symbol01(self, t):
        U = self.U
        sinUt_over_U = np.hstack([ t, np.sin( U[1:]*t ) / U[1:] ])
        return self.symbol00 * sinUt_over_U

    def linear_symbol10(self, t):
        U = self.U
        sinUt_over_U = np.hstack([ t, np.sin( U[1:]*t ) / U[1:] ])
        return self.symbol10 * sinUt_over_U

    def linear_symbol11(self, t):
        U = self.U
        return np.cos(U*t)

    def exp_linear_t(self, u, t):
        eta = u[0]
        u = u[1]
        res0 = self.apply_operator(eta, self.linear_symbol00(t))
        res0 = res0 + self.apply_operator(u, self.linear_symbol01(t))
        res1 = self.apply_operator(eta, self.linear_symbol10(t))
        res1 = res1 + self.apply_operator(u, self.linear_symbol11(t))
        res = np.array([ res0, res1 ])
        return res.real

    def initialize_linear_symbols(self):
        self.linear_symbol = np.empty((2,2), type(self.linear_symbol00))
        self.linear_symbol[0, 0] = self.linear_symbol00
        self.linear_symbol[0, 1] = self.linear_symbol01
        self.linear_symbol[1, 0] = self.linear_symbol10
        self.linear_symbol[1, 1] = self.linear_symbol11

    def symplectic_Euler_step(self, u, h, M = 2):
        v = u + h * self[0,0](u)
        res = v
        for i in range(M):
            v = h * self[0,1](v)
            res = res + v
        res = res + h * self[1,0](res) + h * self[1,1](res)
        return res

    def apply_thHD_over_HD(self, u):
        return self.apply_operator(u, self.thHD_over_HD).real

    def apply_HD_over_thHD(self, u):
        return self.apply_operator(u, self.HD_over_thHD).real

    def apply_sqrt_gD_over_thHD(self, u):
        return self.apply_operator(u, self.sqrt_gD_over_thHD).real

    def differentiate(self, u):
        return -self.apply_operator(u, self.frequencies).imag



class Kalisch(Whitham):
    def __init__(
        self, length, Ngrid, gravity, water_depth
    ):
        super(Kalisch, self).__init__(length, Ngrid, gravity, water_depth)
        D = self.frequencies
        thHD = np.tanh( water_depth * D)
        self.symbol00 = -1j * thHD
        self.symbol01 = -1j * D
        self.symbol10 = -1j * gravity * D
        self.symbol11 = -0.5j * D
        
    def cut_high_frequencies(self, u, Nleft): # move to base.Equation later
        N = round(Nleft/2)
        symbol = np.zeros_like(self.frequencies)
        for i in range(-N, N):
            symbol[i] = 1.
        res0 = self.apply_operator(u[0], symbol).real
        res1 = self.apply_operator(u[1], symbol).real
        return np.array([ res0, res1 ])
        
    def Hamiltonian(self, u):
        eta = u[0]
        u = u[1]
        res = self.depth * u * self.apply_thHD_over_HD(u)
        res = res + eta * ( self.gravity * eta + u**2 )
        return 0.5 * self.integrate(res)



class Hur(Whitham):
    def __init__(
        self, length, Ngrid, gravity, water_depth
    ):
        super(Hur, self).__init__(length, Ngrid, gravity, water_depth)
        D = self.frequencies
        thHD = np.tanh( water_depth * D)
        self.symbol00 = -1j * water_depth * D
        self.symbol01 = -1j * D
        self.symbol10 = -1j * gravity/water_depth * thHD
        self.symbol11 = -0.5j * D



class Hamiltonian_Hur(Whitham):
    def __init__(
        self, length, Ngrid, gravity, water_depth
    ):
        super(Hamiltonian_Hur, self).__init__(length, Ngrid, gravity, water_depth)
        D = self.frequencies
        thHD = np.tanh( water_depth * D)
        self.symbol00 = -1j * water_depth * D
        self.symbol01 = -1j / water_depth * thHD
        self.symbol10 = -1j * gravity/water_depth * thHD
        self.symbol11 = -0.5j / water_depth * thHD
        
    def Hamiltonian(self, u):
        eta = u[0]
        u = u[1]
        res = self.depth * u * self.apply_HD_over_thHD(u)
        res = res + eta * ( self.gravity * eta + u**2 )
        return 0.5 * self.integrate(res)



class Matsuno(Kalisch):
    def __init__(
        self, length, Ngrid, gravity, water_depth
    ):
        super(Matsuno, self).__init__(length, Ngrid, gravity, water_depth)
        self.num_right_terms = np.array([3, 3])
        D = self.frequencies
        thHD = np.tanh( water_depth * D)
        self.thHD = thHD
        self.symbol02 = 1j * D * thHD
        self.symbol12 = self.symbol11
        

    def __getitem__(self, index):
        if index == (0, 2):
            res = self.F02
        elif index == (1, 2):
            res = self.F12
        else:
            res = super(Matsuno, self).__getitem__(index)
        return res

    def F02(self, u, t = 0):
        eta = u[0]
        u = u[1]
        u = self.apply_operator(u, self.thHD)
        res = self.apply_operator(eta*u, self.symbol02)
        res = res.real
        return np.array([ res, np.zeros_like(res) ])
        
    def F12(self, u, t = 0):
        u = u[1]
        u = self.apply_operator(u, self.thHD)
        res = self.apply_operator(u**2, self.symbol12)
        res = res.real
        return np.array([ np.zeros_like(res), res ])
        
    def Hamiltonian(self, u):
        res = super(Matsuno, self).Hamiltonian(u)
        eta = u[0]
        u = u[1]
        u = self.apply_operator(u, self.thHD)
        u = u**2
        u = u.real
        return res + 0.5 * self.integrate(eta * u) 



class RS(Whitham):
    def __init__(
        self, length, Ngrid, gravity, water_depth
    ):
        super(RS, self).__init__(length, Ngrid, gravity, water_depth)
        self.symbol00 = -1j * self.U * np.sign(self.frequencies)
        self.symbol01 = self.symbol00 / (4 * water_depth)
        self.symbol10 = - self.symbol00
        self.symbol11 = self.symbol01
        self.sqrt_thHD_over_gD = 1.0 / self.sqrt_gD_over_thHD

    def F00(self, u, t = 0):
        r = u[0]
        res = self.apply_operator(r, self.symbol00)
        res = res.real
        return np.array([ res, np.zeros_like(res) ])
        
    def F01(self, u, t = 0):
        r = u[0]
        s = u[1]
        res = (3*r + s) * (r - s)
        res = self.apply_operator(res, self.symbol01)
        res = res.real
        return np.array([ res, np.zeros_like(res) ])
        
    def F10(self, u, t = 0):
        s = u[1]
        res = self.apply_operator(s, self.symbol10)
        res = res.real
        return np.array([ np.zeros_like(res), res ])
        
    def F11(self, u, t = 0):
        r = u[0]
        s = u[1]
        res = (r + 3*s) * (r - s)
        res = self.apply_operator(res, self.symbol11)
        res = res.real
        return np.array([ np.zeros_like(res), res ])

    def linear_symbol00(self, t):
        return np.exp(self.symbol00 * t)

    def linear_symbol01(self, t):
        return 0

    def linear_symbol10(self, t):
        return 0

    def linear_symbol11(self, t):
        return np.exp(self.symbol10 * t)

    def symplectic_Euler_step(self):
        raise NotImplementedError()
        
    def Hamiltonian(self, u):
        r = u[0]
        s = u[1]
        res = r**2 + s**2
        res = res + (r + s) * (r - s)**2 / (2 * self.depth)
        return self.gravity * self.integrate(res)
        
    def Hamiltonian_coupling(self, u):
        r = u[0]
        s = u[1]
        res = r * s * (r + s)
        const = - self.gravity / (2 * self.depth)
        return const * self.integrate(res)

    def apply_sqrt_thHD_over_gD(self, u):
        return self.apply_operator(u, self.sqrt_thHD_over_gD).real
        