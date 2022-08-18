from __future__ import division
import numpy as np
#import newton
from travwave.solver import * 


class Solver2d(Solver):
    def __init__(
        self, equation, time_step, sixth_order_splitting = False,
        expA_is_exact = False, expA_is_Runge_Kutta4 = False,
        expB_is_exact = False, expB_is_Runge_Kutta4 = False,
        right_partA_indices = [], right_partB_indices = []
    ):
        super(Solver2d, self).__init__(
        equation, time_step, sixth_order_splitting,
        expA_is_exact, expA_is_Runge_Kutta4,
        expB_is_exact, expB_is_Runge_Kutta4,
        right_partA_indices, right_partB_indices
    )
            
"""
    def exact(self, u, t_ind):
#        One step of exact exp method.
        res = np.zeros_like(u)
        n = self.equation.num_unknowns
        for i in range(n):
            for k in range(n):
                res[i] = res[i] + \
                self.equation.apply_operator(u[k], self.linear_symbol[t_ind, i, k]).real
        return res

    def initialize_linear_symbols(self):
        Ntimes = len(self.times)
        N = self.equation.num_unknowns
        self.linear_symbol = np.empty( (Ntimes, N, N), dtype=np.ndarray )
        for t in range(Ntimes):
            for i in range(N):
                for k in range(N):
                    self.linear_symbol[t, i, k] = self.equation.linear_symbol[i, k](self.times[t])
        
"""        