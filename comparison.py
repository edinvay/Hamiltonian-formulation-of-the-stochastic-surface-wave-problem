from __future__ import division
from matplotlib import pyplot as plt
from pylab import *
import numpy as np
from copy import deepcopy
import os as os_functions
import pickle

from travwave.equations import *
from travwave.solver import *
from travwave import projector
import scipy.io

def calibrate_noise(gamma, epsilon = 0.1, g = 1.0, h = 1.0):
    temp = 2 * np.sqrt(g * h**3) * epsilon / np.sum(gamma**2)
    return gamma * np.sqrt(temp)

def name_solution_file(
    directory_name = 'Experiments',
    file_name = ''
):
    if not os_functions.path.exists(directory_name):
        os_functions.makedirs(directory_name)
    file_name = os_functions.path.join(directory_name, file_name)
    return file_name

def integrate(u, nodes):
    return ( nodes[1] - nodes[0] ) * np.sum(u)

def L2_norm(u, nodes):
    return np.sqrt( integrate(u**2, nodes) )

def distance(solution_euler, solution_whitham, nodes):
    res = []
    for t in range(len(solution_euler)):
        u = solution_euler[t] - solution_whitham[t]
        res.append( L2_norm(u, nodes) )
    return max(res)

def interpolate(Eta, X, nodes):
    return scipy.interpolate.interp1d(X, Eta, fill_value='extrapolate')(nodes)

def solution_interpolation(Eta, X, nodes):
    res = []
    Ngrid, Ntime = np.shape(X)
    for t in range(Ntime):
        res.append( interpolate(Eta[:, t], X[:, t], nodes) )
    return res

def extract_solution(
    directory_name,
    file_name,
    nodes
):
    Solution = dict()
    name = name_solution_file( directory_name, file_name )
    scipy.io.loadmat(name, Solution)
    Eta = solution_interpolation(Solution['Eta'], Solution['X'], nodes)
    return Eta, Solution['Hamiltonian'][0,0]

def extract_eta(solution, system_is_rs = False):
    res = []
    if system_is_rs:
        for sol in solution:
            res.append( sol[0] + sol[1] )
    else:
        for sol in solution:
            res.append( sol[0] )
    return res
