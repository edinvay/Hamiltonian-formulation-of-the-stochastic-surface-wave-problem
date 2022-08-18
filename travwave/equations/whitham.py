from __future__ import division
import scipy.fftpack
#import scipy.interpolate
import numpy as np

from .base import Equation

class Whitham(Equation):
    def __init__(
        self, length, Ngrid, gravity, water_depth
    ):
        num_right_terms = np.array([2])
        super(Whitham, self).__init__(length, Ngrid, 1, num_right_terms)
        self.gravity = gravity
        self.depth = water_depth
        ###
        D = self.frequencies
        thHD = np.tanh( water_depth * D)
        G = D * thHD
        self.symbol00 = -1j * np.sqrt(gravity * G) * np.sign(D)
        self.symbol01 = -1j * 0.75 * np.sqrt(gravity / water_depth) * D
        self.initialize_linear_symbols()

        
    def __getitem__(self, index):    # accessing right part term by index number
        if index == (0, 0):
            res = self.F00
        elif index == (0, 1):
            res = self.F01
        else:
            raise NotImplementedError()
        return res

    def F00(self, u, t = 0):
        res = self.apply_operator(u, self.symbol00)
        return res.real
        
    def F01(self, u, t = 0):
        res = self.apply_operator(u**2, self.symbol01)
        return res.real
        
    def linear_symbol00(self, t):
        return np.exp(self.symbol00 * t)

    def exp_linear_t(self, u, t):
        res = self.apply_operator(u, self.linear_symbol00(t))
        return res.real

    def initialize_linear_symbols(self):
        self.linear_symbol = np.empty((1,1), type(self.linear_symbol00))
        self.linear_symbol[0, 0] = self.linear_symbol00
    


    

class test_Whitham(Whitham):
    def __init__(
        self, length, Ngrid, gravity, water_depth, f, df, g, dg
    ):
        super(test_Whitham, self).__init__(length, Ngrid, gravity, water_depth)
        self.num_right_terms = np.array([3])
        self.f = f(self.nodes)
        self.df = df(self.nodes)
        self.g = g
        self.dg = dg
        ###
        D = self.frequencies
        thHD = np.tanh( water_depth * D)
        symbol02 = np.hstack([ water_depth, thHD[1:] / D[1:] ])
        symbol02 = np.sqrt(gravity * symbol02)
        Wdf = self.apply_operator(self.df, symbol02)
        self.Wdf = Wdf.real
        self.fdf = self.f * self.df
        
        
    def __getitem__(self, index):    # accessing right part term by index number
        if index == (0, 2):
            res = self.F02
        else:
            res = super(test_Whitham, self).__getitem__(index)
        return res

    def F02(self, u, t = 0):
        f = self.f
        fdf = self.fdf
        Wdf = self.Wdf
        g = self.g(t)
        dg = self.dg(t)
        
        res = f * dg + g * Wdf
        res = res + 1.5 * np.sqrt(self.gravity / self.depth) * (g**2) * fdf
        return res
