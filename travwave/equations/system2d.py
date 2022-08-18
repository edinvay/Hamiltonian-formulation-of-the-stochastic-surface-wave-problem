from __future__ import division
import scipy.fftpack
import scipy.interpolate
import numpy as np

from .base import Equation2d

class Ice2d(Equation2d):
    def __init__(
        self, Lx, Nx, Ly, Ny,
        gravity, water_depth, kappa_elasticity, thickness,
        water_density, ice_density, damping
    ):
        num_right_terms = np.array([2, 3])
        super(Ice2d, self).__init__(Lx, Nx, Ly, Ny, 2, num_right_terms)
        self.gravity = gravity
        self.depth = water_depth
        self.kappa = kappa_elasticity
        self.thickness = thickness
        self.density = water_density
        self.ice_density = ice_density
        self.damping = damping
        ###
        Dx = self.xx_freq
        Dy = self.yy_freq
        D = np.sqrt(Dx**2 + Dy**2)
        thHD = np.tanh( water_depth * D)
        G = D * thHD
        self.symbol00 = G
        K = ( 1 + (thickness * D)**2 / 12 ) * G
        K = 1 + K * ice_density * thickness / water_density
        self.K = K
        B = gravity * (1 + kappa_elasticity * D**4) / K
        R = damping * G / (2 * water_density * K)
        self.R = R
        self.symbol10 = - B
        self.symbol11 = -2 * R
        self.U = np.sqrt(G * B - R**2)
##############################################################
#Nonlinear part:
#        self.symbol01 = -1j * D
#        self.symbol12 = -0.5j * D
##############################################################
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
        u = u[1]
        res = self.apply_operator(u, self.symbol00)
        res = res.real
        return np.array([ res, np.zeros_like(res) ])
        
    def F10(self, u, t = 0):
        eta = u[0]
        res = self.apply_operator(eta, self.symbol10)
        res = res.real
        return np.array([ np.zeros_like(res), res ])
        
    def F11(self, u, t = 0):
        u = u[1]
        res = self.apply_operator(u, self.symbol11)
        res = res.real
        return np.array([ np.zeros_like(res), res ])

    def linear_symbol00(self, t):
        R = self.R
        U = self.U
        U[0,0] = 1.0
        sinUt_over_U = np.sin(U*t) / U
        sinUt_over_U[0,0] = t
        U[0,0] = 0.0
        res = R * sinUt_over_U + np.cos(U*t)
        return res * np.exp(-R * t)

    def linear_symbol01(self, t):
        R = self.R
        U = self.U
        U[0,0] = 1.0
        sinUt_over_U = np.sin(U*t) / U
        sinUt_over_U[0,0] = t
        U[0,0] = 0.0
        res = self.symbol00 * sinUt_over_U
        return res * np.exp(-R * t)

    def linear_symbol10(self, t):
        R = self.R
        U = self.U
        U[0,0] = 1.0
        sinUt_over_U = np.sin(U*t) / U
        sinUt_over_U[0,0] = t
        U[0,0] = 0.0
        res = self.symbol10 * sinUt_over_U
        return res * np.exp(-R * t)

    def linear_symbol11(self, t):
        R = self.R
        U = self.U
        U[0,0] = 1.0
        sinUt_over_U = np.sin(U*t) / U
        sinUt_over_U[0,0] = t
        U[0,0] = 0.0
        res = -R * sinUt_over_U + np.cos(U*t)
        return res * np.exp(-R * t)

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
        
##############################################################
#Nonlinear part:
#    def F01(self, u, t = 0):
#        eta = u[ : self.Ngrid]
#        u = u[self.Ngrid : ]
#        res = self.apply_operator(eta*u, self.symbol01)
#        return np.hstack([ res.real, np.zeros(self.Ngrid) ])
        
#    def F12(self, u, t = 0):
#        u = u[self.Ngrid : ]
#        res = self.apply_operator(u**2, self.symbol12)
#        return np.hstack([ np.zeros(self.Ngrid), res.real ])
##############################################################

    


    
class Loaded_ice2d(Ice2d):
    def __init__(
        self, Lx, Nx, Ly, Ny,
        gravity, water_depth, kappa_elasticity, thickness,
        water_density, ice_density, damping,
        load, load_velocity, initial_position = 0.0,
        load_length = None, load_width = None
    ):
        super(Loaded_ice2d, self).__init__(
            Lx, Nx, Ly, Ny,
            gravity, water_depth, kappa_elasticity, thickness,
            water_density, ice_density, damping
        )
        self.num_right_terms = np.array([2, 7])
        self.load_velocity = load_velocity
        self.initial_position = initial_position
        self.initial_index = self.get_x_index(initial_position)
        self.load_length = load_length
        self.load_width = load_width
        ###
        temp = load * (Nx / (2*Lx)) * (Ny / (2*Ly))
        self.normed_load_over_K = temp * self.load_fourier() / self.K
        temp = scipy.fftpack.ifft2( self.normed_load_over_K )
        temp = scipy.fftpack.fftshift(temp)
        self.w = temp.real
        rho_h_rho = ice_density * thickness / water_density
        self.symbol140 = rho_h_rho * self.xx_freq / self.K
        self.symbol141 = rho_h_rho * self.yy_freq / self.K
        self.symbol142 = - rho_h_rho * self.symbol00 / self.K
        ###
        self.x_extended = np.hstack([ self.x_nodes - 2*Lx, self.x_nodes, self.x_nodes + 2*Lx ])
        self.w_extended = np.hstack([ self.w, self.w, self.w ])
        self.initialize_shifted_w()
        self.initialize_Dx_w()
        self.initialize_Dy_w()
        self.initialize_G_w()
        ###
##############################################################
#Nonlinear part:
#        temp = temp * (-0.5j * gravity)
#        self.symbol15 = temp * self.frequencies**3
#        self.symbol16 = (- damping / water_dencity) * self.frequencies**2
##############################################################
        

    def load_fourier(self):
        length_fourier = 1.0
        if self.load_length != None:
            length_fourier = 0.5 * self.load_length * self.xx_freq
            length_fourier[:,0] = np.ones(self.Ny)
            length_fourier = np.sin(length_fourier) / length_fourier
            length_fourier[:,0] = np.ones(self.Ny)
        width_fourier = 1.0
        if self.load_width != None:
            width_fourier = 0.5 * self.load_width * self.yy_freq
            width_fourier[0,:] = np.ones(self.Nx)
            width_fourier = np.sin(width_fourier) / width_fourier
            width_fourier[0,:] = np.ones(self.Nx)
        return length_fourier * width_fourier

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
            res = super(Loaded_ice2d, self).__getitem__(index)
        return res
        
    def F13(self, u, t = 0):
        res = - self.shifted_w[self.get_x_index_by_time(t)]
#        res =  - self.shift_w(x)
        return np.array([ np.zeros_like(res), res ])
        
    def F14(self, u, t = 0):
        eta = u[0]
        ind = self.get_x_index_by_time(t)
        res = self.apply_operator(eta * self.Dx_w[ind], self.symbol140)
        res = res + self.apply_operator(eta * self.Dy_w[ind], self.symbol141)
        res = res + self.apply_operator(eta * self.G_w[ind], self.symbol142)
        res = res.real
        return np.array([ np.zeros_like(res), res ])
    
    def shift_w(self, shift):
        f = scipy.interpolate.interp2d(self.x_extended + shift, self.y_nodes, self.w_extended)
        return f(self.x_nodes, self.y_nodes)

    def initialize_shifted_w(self):
        self.shifted_w = np.empty(self.Nx, type(self.w))
        for i in range(self.Nx):
            self.shifted_w[i] = np.zeros_like(self.w)
#        for i in range(self.Nx):
#            self.shifted_w[i] = self.shift_w(self.x_nodes[i])
        N_over_2 = round(self.Nx/2)
        self.shifted_w[0][ :, : N_over_2 ] = self.w[ :, N_over_2 : ]
        self.shifted_w[0][ :, N_over_2 : ] = self.w[ :, 0 : N_over_2 ]
        for i in range(self.Nx)[1:]:
            self.shifted_w[i][ :, 0 ] = self.shifted_w[i-1][ :, -1 ]
            self.shifted_w[i][ :, 1 :  ] = self.shifted_w[i-1][ :, 0 : self.Nx - 1 ]

    def initialize_Dx_w(self):
        self.Dx_w = np.zeros_like(self.shifted_w)
        for i in range(self.Nx):
            self.Dx_w[i] = self.apply_operator(self.shifted_w[i], self.xx_freq)

    def initialize_Dy_w(self):
        self.Dy_w = np.zeros_like(self.shifted_w)
        for i in range(self.Nx):
            self.Dy_w[i] = self.apply_operator(self.shifted_w[i], self.yy_freq)

    def initialize_G_w(self):
        self.G_w = np.zeros_like(self.shifted_w)
        for i in range(self.Nx):
            self.G_w[i] = self.apply_operator(self.shifted_w[i], self.symbol00)

    def get_x_index(self, x):
        ind = 0
        for i in range(self.Nx-1):
            if self.x_nodes[i] <= x and x < self.x_nodes[i+1]:
                ind = i
                if self.x_nodes[i+1] - x < x - self.x_nodes[i]:
                    ind = i + 1
        return ind    

    def get_x_index_by_time(self, t):
        x = self.initial_position + self.load_velocity * t
        return self.get_x_index(x)
    
    def shift_by_index(self, f, ind):
        res = np.empty_like(f)
        N_over_2 = round(self.Nx/2)
        for i in range( -N_over_2, N_over_2 ):
            res[:, i] = f[:, N_over_2 - ind + i]
        return res
    
    
    
class Loaded_remembering_ice2d(Loaded_ice2d):
    def __init__(
        self, Lx, Nx, Ly, Ny,
        gravity, water_depth, kappa_elasticity, thickness,
        water_density, ice_density, damping,
        load, load_velocity, initial_position = 0.0,
        load_length = None, load_width = None,
        memory_A = 0.1, memory_alpha = 0.1,
        nontrivial_memory = False,
        previous_eta = None,
        time_step = 0.1
    ):
        super(Loaded_remembering_ice2d, self).__init__(
        Lx, Nx, Ly, Ny,
        gravity, water_depth, kappa_elasticity, thickness,
        water_density, ice_density, damping,
        load, load_velocity, initial_position,
        load_length, load_width
        )
        self.num_right_terms = np.array([2, 8])
        self.memory_A = memory_A
        self.memory_alpha = memory_alpha
        if nontrivial_memory == False:  #previous_eta == None
            previous_eta = np.zeros_like(self.xx)
        self.memory_integral = memory_A / memory_alpha * previous_eta
        self.memory_symbol = - self.symbol10 - self.gravity / self.K
        self.memory = None
        self.calculate_memory()
        self.previous_eta = previous_eta
        self.exp_minus_alpha_h = None
        self.hA_over_2 = None
        self.calculate_time_step_papameters(time_step)
    
    def calculate_memory(self):
        self.memory = self.apply_operator(self.memory_integral, self.memory_symbol).real
    
    def calculate_time_step_papameters(self, time_step):
        self.exp_minus_alpha_h = np.exp(- self.memory_alpha * time_step)
        self.hA_over_2 = time_step * self.memory_A / 2
    
    def modify_memory(self, solution):
        temp = self.previous_eta * self.exp_minus_alpha_h + solution
        temp = temp * self.hA_over_2
        self.memory_integral = self.memory_integral * self.exp_minus_alpha_h + temp
        self.calculate_memory()
        self.previous_eta = solution
    
    def __getitem__(self, index):    # accessing right part term by index number
        if index == (1, 7):
            res = self.F17
        else:
            res = super(Loaded_remembering_ice2d, self).__getitem__(index)
        return res
        
    def F17(self, u, t = 0):
        return np.array([ np.zeros_like(self.memory), self.memory ])
        
    
    
    
"""


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
    
    

    
    
    

"""