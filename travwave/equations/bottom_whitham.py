from __future__ import division
#import scipy.fftpack
#import scipy.interpolate
import numpy as np

from .bidirectional_whitham import *



class Hamiltonian_Hur_bottom(Hamiltonian_Hur):
    def __init__(
        self, length, Ngrid, gravity, water_depth, bottom, order = 1
    ):
        super(Hamiltonian_Hur_bottom, self).__init__(length, Ngrid, gravity, water_depth)
        self.bottom = bottom
        D = self.frequencies
        thHD = np.tanh( water_depth * D)
        self.sechHD = 1. / np.cosh( water_depth * D)
        self.D_thHD = D * thHD
        self.apply_LinvD = self.assign_LinvD(order)
        
        

    def __getitem__(self, index):
        if index == (0, 2):
            res = self.F02
        elif index == (1, 2):
            res = self.F12
        elif index == (1, 3):
            res = self.F13
        elif index == (0, 3):
            res = self.F03
        elif index == (0, 4):
            res = self.F04
        elif index == (0, 5):
            res = self.F05
        else:
            res = super(Hamiltonian_Hur_bottom, self).__getitem__(index)
        return res

    def F02(self, u, t = 0):
        u = u[1]
        u = self.apply_HD_over_thHD(u)
        u = self.apply_LinvD(u)
        res = self.apply_operator(u, self.symbol00)
        res = res.real / self.depth
        return np.array([ res, np.zeros_like(res) ])

    def F12(self, u, t = 0):
        res = self.F02(u)
        res = self.F11( [res[1], res[0]] )
        return -res 

    def F13(self, u, t = 0):
        res1 = self.F00(u)[0]
        res2 = self.F02(u)
        res1 = self.F01( [res1, res2[0]] )
        res1[1] = -res1[0]
        res1[0] = res2[1]
        return res1 

    def F03(self, u, t = 0):
        res = self.F02(u)[0]
        res = self.F01( [u[0], res] )[0]
        res = self.F02( [u[0], res] )
        return res

    def F04(self, u, t = 0):
        res = self.F00(u)[0]
        res = self.F01( [u[0], res] )[0]
        res = self.F02( [u[0], res] )
        return res

    def F05(self, u, t = 0):
        res = self.F02(u)[0]
        res = self.F01( [u[0], res] )[0]
        res = self.F00( [u[0], res] )
        return res

    
    def Hamiltonian(self, u):
        res = super(Hamiltonian_Hur_bottom, self).Hamiltonian(u)
        u = u[1]
        u = self.apply_HD_over_thHD(u)
        u = u * self.apply_LinvD(u)
        return res + 0.5 * self.integrate(u) 

    def assign_LinvD(self, order):
        if order == 1:
            res = self.apply_LinvD1
        elif order == 2:
            res = self.apply_LinvD2
        else:
            raise NotImplementedError()
        return res 

    def apply_LinvD1(self, u):
        beta_sech = self.bottom * self.apply_sechHD(u)
        res = self.apply_sechHD(beta_sech)
        return -res

    def apply_LinvD2(self, u):
        beta_sech = self.bottom * self.apply_sechHD(u)
        res = self.bottom * self.apply_D_thHD(beta_sech)
        res = res + beta_sech
        res = self.apply_sechHD(res)
        return -res

#    def apply_LinvD3(self, u):
#        beta_sech = self.bottom * self.apply_sechHD(u)
#        res = self.apply_sechHD(beta_sech)
#        return -res

    def apply_sechHD(self, u):
        return self.apply_operator(u, self.sechHD).real

    def apply_D_thHD(self, u):
        return self.apply_operator(u, self.D_thHD).real
    