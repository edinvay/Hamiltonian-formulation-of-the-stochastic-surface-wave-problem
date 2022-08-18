from __future__ import division
from copy import deepcopy
import scipy.fftpack
import numpy as np

class Equation(object):
    def __init__(self, length, Ngrid, num_unknowns, num_right_terms):
        self.length = length
        self.Ngrid = Ngrid
        self.nodes = length * np.linspace(-1, 1, Ngrid, endpoint=False)
        self.frequencies = scipy.fftpack.fftfreq(Ngrid, length / (np.pi * Ngrid))
        self.num_unknowns = num_unknowns
        self.num_right_terms = deepcopy(num_right_terms)

    def __getitem__(self, index):    # accessing right part term by index number
        raise NotImplementedError()

    def apply_operator(self, u, symbol):     # returns operator applied to u
                                             # InverseFourier( symbol(k)* Fourier(u)(k) )
        u_ = scipy.fftpack.fft(u)
        return scipy.fftpack.ifft(symbol * u_)

    def integrate(self, u):
        return ( self.nodes[1] - self.nodes[0] ) * np.sum(u)


class Equation2d(object):
    def __init__(self, Lx, Nx, Ly, Ny, num_unknowns, num_right_terms):
        self.Lx = Lx
        self.Nx = Nx
        self.Ly = Ly
        self.Ny = Ny
        ###
        self.x_nodes = Lx * np.linspace(-1, 1, Nx, endpoint=False)
        self.x_frequencies = scipy.fftpack.fftfreq(Nx, Lx / (np.pi * Nx))
        self.y_nodes = Ly * np.linspace(-1, 1, Ny, endpoint=False)
        self.y_frequencies = scipy.fftpack.fftfreq(Ny, Ly / (np.pi * Ny))
        ###
        self.xx, self.yy = np.meshgrid(self.x_nodes, self.y_nodes)
        self.xx_freq, self.yy_freq = np.meshgrid(self.x_frequencies, self.y_frequencies)
        ###
        self.num_unknowns = num_unknowns
        self.num_right_terms = deepcopy(num_right_terms)

    def __getitem__(self, index):    # accessing right part term by index number
        raise NotImplementedError()

    def apply_operator(self, u, symbol):     # returns operator applied to u
                                             # InverseFourier( symbol(k)* Fourier(u)(k) )
        u_ = scipy.fftpack.fft2(u)
        return scipy.fftpack.ifft2(symbol * u_)
