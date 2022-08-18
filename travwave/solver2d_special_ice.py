from __future__ import division
import numpy as np
from travwave.solver import *


class Solver2d_special_ice(Solver):
    def __init__(
        self, equation, time_step, sixth_order_splitting = False,
        expA_is_exact = False, expA_is_Runge_Kutta4 = False,
        expB_is_exact = False, expB_is_Runge_Kutta4 = False,
        right_partA_indices = [], right_partB_indices = [],
        ###
        expB_is_special = False, ZERO_SPEED = 0.0001
    ):
        super(Solver2d_special_ice, self).__init__(
        equation, time_step, sixth_order_splitting,
        expA_is_exact, expA_is_Runge_Kutta4,
        expB_is_exact, expB_is_Runge_Kutta4,
        right_partA_indices, right_partB_indices
    )
        self.load_is_moving = True
        if abs(self.equation.load_velocity) < ZERO_SPEED:
            self.load_is_moving = False
        if expB_is_special == True:
            self.expB = self.exp_by_special
        self.initialize_special_symbols()    

        
    def exp_by_special(self, u, t, ind):
        return self.special(u, t, ind)
    
    def special(self, u, t, ind):
        u = u + self.difference_due_to_outer_presure(u, t, ind)
        u[1] = u[1] + self.difference_due_to_memory(u[0], ind)
        return u
        
    def difference_due_to_outer_presure(self, u, t, ind):
        res = self.equation.F13(u, t)
        res[1] = res[1] + self.equation.F14(u, t)[1]
        res[1] = res[1] * self.times[ind]
        return res   #res[0] = 0
        
    def difference_due_to_memory(self, eta, ind):
        res = self.equation.apply_operator(eta, self.memory_symbols[ind])
        res = res.real
        res = res + self.equation.memory * self.memory_weights[ind]
        return res

    def initialize_special_symbols(self):
        alpha = self.equation.memory_alpha
        A = self.equation.memory_A
        alpha_s = alpha * self.times
        self.memory_weights = ( 1.0 - np.exp( - alpha_s ) ) / alpha
        ###
        Ntimes = len(self.times)
        temp = (A / alpha**2) * ( np.exp( - alpha_s ) + alpha_s - 1.0 )
        self.memory_symbols = np.empty( Ntimes, dtype=np.ndarray )
        for t in range(Ntimes):
            self.memory_symbols[t] = self.equation.memory_symbol * temp[t]
        
"""
    def difference_due_to_outer_presure(self, u, t, ind):
        if False:
            res = self.equation.F13(u, t) + self.equation.F14(u, t)
            res = res - self.equation.F13(u, t + self.times[ind])
            res = res - self.equation.F14(u, t + self.times[ind])
            res[1] = self.equation.apply_operator(res[1], inv_vdx_symbol).real
        else:
#            res = self.equation.F13(u) + self.equation.F14(u)
            res = self.equation.F13(u, t) + self.equation.F14(u, t)
            res[1] = res[1] * self.times[ind]
        return res
        
    def initialize_special_symbols(self):
#        if self.load_is_moving:
        if False:
            inv_vdx_symbol = -1j / ( self.equation.load_velocity * self.equation.xx_freq )
"""