from __future__ import division
import scipy.fftpack
import scipy.interpolate
import numpy as np

from .base import Equation

class Ice(Equation):
    def __init__(
        self, length, Ngrid, gravity, water_depth, kappa_elasticity, thickness,
        water_dencity, ice_density, damping
    ):
        num_right_terms = np.array([2, 3])
        super(Ice, self).__init__(length, Ngrid, 2, num_right_terms)
        self.gravity = gravity
        self.depth = water_depth
        self.kappa = kappa_elasticity
        self.thickness = thickness
        self.water_dencity = water_dencity
        self.ice_density = ice_density
        self.damping = damping
        ###
        D = self.frequencies
        thHD = np.tanh( water_depth * D)
        self.symbol00 = -1j * thHD
        self.symbol01 = -1j * D
        G = D * thHD
        K = ( 1 + (thickness * D)**2 / 12 ) * G
        K = 1 + K * ice_density * thickness / water_dencity
        self.K = K
        B = gravity * (D + kappa_elasticity * D**5) / K
        R = damping * G / (2 * water_dencity * K)
        self.R = R
        self.symbol10 = -1j * B
        self.symbol11 = -2 * R
        self.symbol12 = -0.5j * D
        self.U = np.sqrt(thHD * B - R**2)
        self.initialize_linear_symbols()

        
    def __getitem__(self, index):    # accessing right part term by index number
        if index == (0, 0):
            res = self.F00
        elif index == (0, 1):
            res = self.F01
        elif index == (1, 0):
            res = self.F10
        elif index == (1, 1):
            res = self.F11
        elif index == (1, 2):
            res = self.F12
        else:
            raise NotImplementedError()
        return res

    def F00(self, u, t = 0):
        u = u[self.Ngrid : ]
        res = self.apply_operator(u, self.symbol00)
        return np.hstack([ res.real, np.zeros(self.Ngrid) ])
        
    def F01(self, u, t = 0):
        eta = u[ : self.Ngrid]
        u = u[self.Ngrid : ]
        res = self.apply_operator(eta*u, self.symbol01)
        return np.hstack([ res.real, np.zeros(self.Ngrid) ])
        
    def F10(self, u, t = 0):
        eta = u[ : self.Ngrid]
        res = self.apply_operator(eta, self.symbol10)
        return np.hstack([ np.zeros(self.Ngrid), res.real ])
        
    def F11(self, u, t = 0):
        u = u[self.Ngrid : ]
        res = self.apply_operator(u, self.symbol11)
        return np.hstack([ np.zeros(self.Ngrid), res.real ])
        
    def F12(self, u, t = 0):
        u = u[self.Ngrid : ]
        res = self.apply_operator(u**2, self.symbol12)
        return np.hstack([ np.zeros(self.Ngrid), res.real ])

    def linear_symbol00(self, t):
        R = self.R
        U = self.U
        sinUt_over_U = np.hstack([ t, np.sin( U[1:]*t ) / U[1:] ])
        res = R * sinUt_over_U + np.cos(U*t)
        return res * np.exp(-R * t)

    def linear_symbol01(self, t):
        R = self.R
        U = self.U
        sinUt_over_U = np.hstack([ t, np.sin( U[1:]*t ) / U[1:] ])
        res = self.symbol00 * sinUt_over_U
        return res * np.exp(-R * t)

    def linear_symbol10(self, t):
        R = self.R
        U = self.U
        sinUt_over_U = np.hstack([ t, np.sin( U[1:]*t ) / U[1:] ])
        res = self.symbol10 * sinUt_over_U
        return res * np.exp(-R * t)

    def linear_symbol11(self, t):
        R = self.R
        U = self.U
        sinUt_over_U = np.hstack([ t, np.sin( U[1:]*t ) / U[1:] ])
        res = -R * sinUt_over_U + np.cos(U*t)
        return res * np.exp(-R * t)

    def exp_linear_t(self, u, t):
        eta = u[ : self.Ngrid]
        u = u[self.Ngrid : ]
        res0 = self.apply_operator(eta, self.linear_symbol00(t))
        res0 = res0 + self.apply_operator(u, self.linear_symbol01(t))
        res1 = self.apply_operator(eta, self.linear_symbol10(t))
        res1 = res1 + self.apply_operator(u, self.linear_symbol11(t))
        res = np.hstack([ res0, res1 ])
        return res.real

    def initialize_linear_symbols(self):
        self.linear_symbol = np.empty((2,2), type(self.linear_symbol00))
        self.linear_symbol[0, 0] = self.linear_symbol00
        self.linear_symbol[0, 1] = self.linear_symbol01
        self.linear_symbol[1, 0] = self.linear_symbol10
        self.linear_symbol[1, 1] = self.linear_symbol11
    


    
class Loaded_ice(Ice):
    def __init__(
        self, length, Ngrid, gravity, water_depth, kappa_elasticity, thickness,
        water_dencity, ice_density, damping,
        load, load_velocity, initial_position = 0.0, load_length = None
    ):
        super(Loaded_ice, self).__init__(
            length, Ngrid, gravity, water_depth, kappa_elasticity, thickness,
            water_dencity, ice_density, damping
        )
        self.num_right_terms = np.array([2, 7])
        self.load_velocity = load_velocity
        self.initial_position = initial_position
        load_fourier = 1.0
        if load_length != None:
            load_fourier = 0.5 * load_length * self.frequencies
            load_fourier = np.hstack([ 1.0, np.sin(load_fourier[1:]) / load_fourier[1:] ])
        temp = load * Ngrid / (2*length)
        temp = temp * scipy.fftpack.ifft(self.frequencies / self.K * load_fourier)
        temp = -1j * scipy.fftpack.fftshift(temp)
        self.minus_iw = temp.real
        temp = -ice_density * thickness / water_dencity
        self.symbol14 = temp * self.frequencies**2 / self.K
        self.nodes_extended = np.hstack([ self.nodes - 2*length, self.nodes, self.nodes + 2*length])
        self.minus_iw_extended = np.hstack([ self.minus_iw, self.minus_iw, self.minus_iw])
        ###
        temp = temp * (-0.5j * gravity)
        self.symbol15 = temp * self.frequencies**3
        self.symbol16 = (- damping / water_dencity) * self.frequencies**2
        

    def __getitem__(self, index):    # accessing right part term by index number
        if index == (1, 3):
            res = self.F13
        elif index == (1, 4):
            res = self.F14
        elif index == (1, 5):
            res = self.F15
        elif index == (1, 6):
            res = self.F16
        else:
            res = super(Loaded_ice, self).__getitem__(index)
        return res
        
    def F13(self, u, t = 0):
        res =  self.shift_minus_iw(self.initial_position + self.load_velocity * t)
        return np.hstack([ np.zeros(self.Ngrid), res ])
        
    def F14(self, u, t = 0):
        eta = u[ : self.Ngrid]
        res = self.apply_operator(
#####################################        
                                                                                #error:self.symbol01    
            eta * self.shift_minus_iw(self.initial_position + self.load_velocity * t), self.symbol14
        )
        return np.hstack([ np.zeros(self.Ngrid), res.real ])
    
    def shift_minus_iw(self, shift):
        return scipy.interpolate.interp1d(self.nodes_extended + shift, self.minus_iw_extended)(self.nodes)
#####################################        
    def F15(self, u, t = 0):
        eta = u[ : self.Ngrid]
        res = self.apply_operator(eta**2, self.symbol15)
        return np.hstack([ np.zeros(self.Ngrid), res.real ])
        
    def F16(self, u, t = 0):
        eta = u[ : self.Ngrid]
        u = u[self.Ngrid : ]
        res = self.apply_operator(eta*u, self.symbol16)
        return np.hstack([ np.zeros(self.Ngrid), res.real ])
#####################################        
    


    

class test_Loaded_ice(Loaded_ice):
    def __init__(
        self, length, Ngrid, gravity, water_depth, kappa_elasticity, thickness,
        water_dencity, ice_density, damping,
        load, load_velocity, initial_position,
        f0, g0, dg0,
        f1, g1, dg1
    ):
        super(test_Loaded_ice, self).__init__(
            length, Ngrid, gravity, water_depth, kappa_elasticity, thickness,
            water_dencity, ice_density, damping,
            load, load_velocity, initial_position
        )
        self.num_right_terms = np.array([3, 8])
        self.f0 = f0(self.nodes)
        self.g0 = g0
        self.dg0 = dg0
        self.f1 = f1(self.nodes)
        self.g1 = g1
        self.dg1 = dg1
        

    def __getitem__(self, index):    # accessing right part term by index number
        if index == (0, 2):
            res = self.F02
        elif index == (1, 7):
            res = self.F17
        else:
            res = super(test_Loaded_ice, self).__getitem__(index)
        return res

    def F02(self, u, t = 0):
        f0 = self.f0
        eta = f0 * self.g0(t)
        f1 = self.f1
        u = f1 * self.g1(t)
        Y = np.hstack([ eta, u ])
        dg0 = self.dg0(t)
        res = np.hstack([ f0 * dg0, np.zeros(self.Ngrid) ])
        for i in range(2):
            res = res - self[0, i](Y)
        return res

    def F17(self, u, t = 0):
        f0 = self.f0
        eta = f0 * self.g0(t)
        f1 = self.f1
        u = f1 * self.g1(t)
        Y = np.hstack([ eta, u ])
        dg1 = self.dg1(t)
        res = np.hstack([ np.zeros(self.Ngrid), f1 * dg1 ])
        for i in range(7):
            res = res - self[1, i](Y)
        return res
    
    

    
    
    

#    class LoadTooFast(Exception):
        """
        Exception raised when the load is too fast for chosen length L.
        """