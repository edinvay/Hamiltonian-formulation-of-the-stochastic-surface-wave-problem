#Does NOT help unfortunately((


#import numpy as np
#from __future__ import division
from travwave.equations import *

def solve_equation(
    state_load_solution, time,
    Lx, Nx, Ly, Ny,
    gravity, water_depth, kappa_elasticity, thickness,
    water_density, ice_density, damping,
    load, load_velocity, init_position,
    load_length, load_width
):
    eq = loaded_damping_ice2d.Simple_ice2d(
        Lx, Nx, Ly, Ny,
        gravity, water_depth, kappa_elasticity, thickness,
        water_density, ice_density, damping,
        load, load_velocity, init_position,
        load_length, load_width
    )
    return eq.exp_linear_w_t(state_load_solution, time)
