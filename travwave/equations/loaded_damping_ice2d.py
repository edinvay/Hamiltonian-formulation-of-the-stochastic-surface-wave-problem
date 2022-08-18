from __future__ import division
import scipy.fftpack
import scipy.interpolate
import numpy as np

from .system2d import Loaded_ice2d

class Simple_ice2d(Loaded_ice2d):
    def __init__(
        self, Lx, Nx, Ly, Ny,
        gravity, water_depth, kappa_elasticity, thickness,
        water_density, ice_density, damping,
        load, load_velocity, initial_position = 0.0,
        load_length = None, load_width = None
    ):
        super(Simple_ice2d, self).__init__(
            Lx, Nx, Ly, Ny,
            gravity, water_depth, kappa_elasticity, thickness,
            water_density, ice_density, damping,
            load, load_velocity, initial_position,
            load_length, load_width
        )
        self.R_add_iU = self.R + 1j * self.U
        self.R_minus_iU = self.R - 1j * self.U
        self.minus_ixi1v = - 1j * self.xx_freq * load_velocity
        self.A_eta = None
        self.B_eta = None
        self.C_eta = None
        self.A_Phi = None
        self.B_Phi = None
        self.C_Phi = None
        self.initialize_ABC_functions()
        
        
    def initialize_ABC_functions(self):
        Adenominator = 2j * self.U * ( self.R_add_iU + self.minus_ixi1v )
        Adenominator[0,0] = 1.0
        invAdenominator = 1.0 / Adenominator
        Bdenominator = 2j * self.U * ( self.R_minus_iU + self.minus_ixi1v )
        Bdenominator[0,0] = 1.0
        invBdenominator = 1.0 / Bdenominator
        Cdenominator = ( self.R_add_iU + self.minus_ixi1v ) * ( self.R_minus_iU + self.minus_ixi1v )
        Cdenominator[0,0] = 1.0
        invCdenominator = 1.0 / Cdenominator
        ###
        temp = self.normed_load_over_K * self.symbol00
        self.A_eta = - temp * invAdenominator
        self.B_eta = temp * invBdenominator
        self.C_eta = - temp * invCdenominator
        ###
        self.A_Phi = self.normed_load_over_K * self.R_add_iU * invAdenominator
        self.B_Phi = - self.normed_load_over_K * self.R_minus_iU * invBdenominator
        self.C_Phi = - self.normed_load_over_K * self.minus_ixi1v * invCdenominator
        
    def eta_w(self, t):
        res = self.A_eta * np.exp(-t * self.R_add_iU)
        res = res + self.B_eta * np.exp(-t * self.R_minus_iU)
        res = res + self.C_eta * np.exp(t * self.minus_ixi1v)
        res = scipy.fftpack.ifft2( res )
        res = scipy.fftpack.fftshift(res)
        res = res.real
        res = self.shift_by_index(res, self.initial_index)
        return res
        
    def Phi_w(self, t):
        res = self.A_Phi * np.exp(-t * self.R_add_iU)
        res = res + self.B_Phi * np.exp(-t * self.R_minus_iU)
        res = res + self.C_Phi * np.exp(t * self.minus_ixi1v)
        res[0,0] = -t
        res = scipy.fftpack.ifft2( res )
        res = scipy.fftpack.fftshift(res)
        res = res.real
        res = self.shift_by_index(res, self.initial_index)
        return res

    def exp_linear_w_t(self, u, t):
        return self.exp_linear_t(u, t) + np.array([ self.eta_w(t), self.Phi_w(t) ])

    def alter_load_velocity(self, velocity):
        self.load_velocity = velocity
        self.minus_ixi1v = - 1j * self.xx_freq * velocity
        self.initialize_ABC_functions()
        

#    def __del__(self):
#        del self.R_add_iU
#        del self.R_minus_iU
#        del self.minus_ixi1v
#        del self.A_eta
#        del self.B_eta
#        del self.C_eta
#        del self.A_Phi
#        del self.B_Phi
#        del self.C_Phi