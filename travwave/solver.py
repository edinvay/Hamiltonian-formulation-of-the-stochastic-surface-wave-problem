from __future__ import division
import numpy as np
#import newton


class Solver(object):
    def __init__(
        self, equation, time_step, sixth_order_splitting = False,
        expA_is_exact = False, expA_is_Runge_Kutta4 = False,
        expB_is_exact = False, expB_is_Runge_Kutta4 = False,
        expA_is_Euler = False, expB_is_Euler = False,
        right_partA_indices = [], right_partB_indices = []
    ):
        self.equation = equation
        self.time_step = time_step
        self.times = np.array([time_step])
        self.right_partA_indices = right_partA_indices
        self.right_partB_indices = right_partB_indices
        self.Runge_Kutta4 = self.Runge_Kutta4_normal
        if sixth_order_splitting == True:
            self.initialize_times_for_sixth_order_operator_splitting()
            self.run = self.sixth_order_operator_splitting
            self.Runge_Kutta4 = self.Runge_Kutta4_split
        if expA_is_Runge_Kutta4 == True:
            self.expA = self.expA_by_Runge_Kutta4
        if expB_is_Runge_Kutta4 == True:
            self.expB = self.expB_by_Runge_Kutta4
        if expA_is_exact == True:
            self.expA = self.exp_by_exact
        if expB_is_exact == True:
            self.expB = self.exp_by_exact
        self.initialize_linear_symbols()
        if expA_is_Euler:
            self.expA = self.expA_by_Euler
        if expB_is_Euler:
            self.expB = self.expB_by_Euler


    def sixth_order_operator_splitting(self, u, t = 0):
        """
        Combine soutions of problems u_t = Au and u_t = Bu
        to get solution of problem u_t = Au + Bu of 6-th order accuracy.
        """
        u = self.expA(u, t, 0)  #( w3/2*dt );
        u = self.expB(u, t, 1)  #( w3*dt );
        u = self.expA(u, t, 2)  #( (w3+w2)/2*dt );
        u = self.expB(u, t, 3)  #( w2*dt );
        u = self.expA(u, t, 4)  #( (w2+w1)/2*dt );
        u = self.expB(u, t, 5)  #( w1*dt );
        u = self.expA(u, t, 6)  #( (w1+w0)/2*dt );
        u = self.expB(u, t, 7)  #( w0*dt );
        u = self.expA(u, t, 6)  #( (w0+w1)/2*dt );
        u = self.expB(u, t, 5)  #( w1*dt );
        u = self.expA(u, t, 4)  #( (w1+w2)/2*dt );
        u = self.expB(u, t, 3)  #( w2*dt );
        u = self.expA(u, t, 2)  #( (w2+w3)/2*dt );
        u = self.expB(u, t, 1)  #( w3*dt );
        u = self.expA(u, t, 0)  #( w3/2*dt );
        return u

    def initialize_times_for_sixth_order_operator_splitting(self):
        w3 = 0.7845136104775600
        w2 = 0.2355732133593570
        w1 = -1.177679984178870
        w0 = 1.3151863206839060
        self.times = np.zeros(8)
        self.times[0] = w3 / 2         * self.time_step
        self.times[1] = w3             * self.time_step
        self.times[2] = (w2 + w3) / 2  * self.time_step
        self.times[3] = w2             * self.time_step
        self.times[4] = (w1 + w2) / 2  * self.time_step
        self.times[5] = w1             * self.time_step
        self.times[6] = (w0 + w1) / 2  * self.time_step
        self.times[7] = w0             * self.time_step

    def Runge_Kutta4_normal(self, u, t, h, f):
        """
        One step of Runge-Kutta method of order 4.
        """
        F1 = h * f(u, t)
        F2 = h * f(u + 0.5 * F1, t + 0.5 * h) 
        F3 = h * f(u + 0.5 * F2, t + 0.5 * h)
        F4 = h * f(u + F3, t + h)              
        u = u + (F1 + 2 * F2 + 2 * F3 + F4) / 6.0
        return u

    def Runge_Kutta4_split(self, u, t, h, f):
        """
        One step of Runge-Kutta method of order 4.
        """
        F1 = h * f(u, t)
#################################################        
#Make semi autonomus for a splitting method
#################################################        
        F2 = h * f(u + 0.5 * F1, t)   # + 0.5 * h
        F3 = h * f(u + 0.5 * F2, t)   # + 0.5 * h
        F4 = h * f(u + F3, t)         # + h              
#################################################        
        u = u + (F1 + 2 * F2 + 2 * F3 + F4) / 6.0
        return u

    def Euler(self, u, t, h, f):
        return u + h * f(u, t)

    def right_partA(self, u, t):
        res = 0.
        for ind in self.right_partA_indices:
            res = res + self.equation[ind](u, t)
        return res

    def right_partB(self, u, t):
        res = 0.
        for ind in self.right_partB_indices:
            res = res + self.equation[ind](u, t)
        return res
    
    def expA_by_Runge_Kutta4(self, u, t, ind):
        return self.Runge_Kutta4(u, t, self.times[ind], self.right_partA)
    
    def expB_by_Runge_Kutta4(self, u, t, ind):
        return self.Runge_Kutta4(u, t, self.times[ind], self.right_partB)
    
    def expA_by_Euler(self, u, t, ind):
        return self.Euler(u, t, self.times[ind], self.right_partA)
    
    def expB_by_Euler(self, u, t, ind):
        return self.Euler(u, t, self.times[ind], self.right_partB)
    
    def exp_by_exact(self, u, t, ind):
        return self.exact(u, ind)
            

    def exact(self, u, t_ind):
        """
        One step of exact exp method.
        """
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
#################################################        
#These methods can be used when solutions of systems combined by np.stack()
#################################################        
    def exact(self, u, t_ind):
        #One step of exact exp method.
        res = np.zeros_like(u)
        n = self.equation.num_unknowns
        N = self.equation.Ngrid
        iN = 0
        for i in range(n):
            iN1 = iN + N
            kN = 0
            for k in range(n):
                kN1 = kN + N
                res[iN:iN1] = res[iN:iN1] + \
                self.equation.apply_operator(u[kN:kN1], self.linear_symbol[t_ind, i, k]).real
                kN = kN1
            iN = iN1
        return res

    def initialize_linear_symbols(self):
        Ntimes = len(self.times)
        N = self.equation.num_unknowns
        self.linear_symbol = np.empty( (Ntimes, N, N), type(self.equation.frequencies) )
        for t in range(Ntimes):
            for i in range(N):
                for k in range(N):
                    self.linear_symbol[t, i, k] = self.equation.linear_symbol[i, k](self.times[t])
"""