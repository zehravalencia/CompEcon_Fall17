# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:28:20 2017

@author: zcagi
"""


# import packages
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numba
import scipy.integrate as integrate
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import norm

# given parameters
rho = 0.7605
mu = 0.0
sigma_eps = 0.213
alpha_k = 0.297
alpha_l = 0.650
delta = 0.154
psi = 1.08
r = 0.04
h = 6.616
betafirm = (1 / (1 + r))
w = 0.67


#PRODUCTIVITY SHOCKS

numberofdraws = 80000
eps = np.random.normal(0.0, sigma_eps, size=(numberofdraws))


z = np.empty(numberofdraws)
z[0] = 0.0 + eps[0]
for i in range(1, numberofdraws):
    z[i] = rho * z[i - 1] + (1 - rho) * mu + eps[i]
    
sigma_z = sigma_eps / ((1 - rho ** 2) ** (1 / 2))

#number of grid points is 9
N = 9  # number of grid points
z_cutoffs = (sigma_z * norm.ppf(np.arange(N + 1) / N)) + mu


z_grid = np.exp((((N * sigma_z * (norm.pdf((z_cutoffs[:-1] - mu) / sigma_z)
                              - norm.pdf((z_cutoffs[1:] - mu) / sigma_z)))
              + mu))) #we calculate grid points for z by taking into account logarithm
    
    
# using ADDA-COPPER method we  compute transition probabilities   
def integrand(x, sigma_z, sigma_eps, rho, mu, z_j, z_jp1):
    val = (np.exp((-1 * ((x - mu) ** 2)) / (2 * (sigma_z ** 2)))
            * (norm.cdf((z_jp1 - (mu * (1 - rho)) - (rho * x)) / sigma_eps)
               - norm.cdf((z_j - (mu * (1 - rho)) - (rho * x)) / sigma_eps)))

    return val


pi = np.empty((N, N))
for i in range(N):
    for j in range(N):
        results = integrate.quad(integrand, z_cutoffs[i], z_cutoffs[i + 1],
                                 args = (sigma_z, sigma_eps, rho, mu,
                                         z_cutoffs[j], z_cutoffs[j + 1]))
        pi[i,j] = (N / np.sqrt(2 * np.pi * sigma_z ** 2)) * results[0]



#DEFINING FUNCTIONS

#FUNCTION FOR VALUE FUNCTION LOOP
@numba.jit
def VFI_loop(V, e, betafirm, sizez, sizek, Vmat, pi):
    V_prime = np.dot(pi, V)
    for i in range(sizez):  # loop over z
        for j in range(sizek):  # loop over k
            for k in range(sizek): # loop over k'
                Vmat[i, j, k] = e[i, j, k] + betafirm * V_prime[i, k]
    return Vmat



# FUNCTION FOR STATIONARY DISTRIBUTION LOOP 
@numba.jit
def SD_loop(PF, pi, Gamma, sizez, sizek):
    HGamma = np.zeros((sizez, sizek))
    for i in range(sizez):  # z
        for j in range(sizek):  # k
            for m in range(sizez):  # z'
                HGamma[m, PF[i, j]] = \
                    HGamma[m, PF[i, j]] + pi[i, m] * Gamma[i, j]
    return HGamma


# FUNCTION TO CALCULATE AGGREGATE VALUES

### Define functions
def aggregate(matrixfirst, matrixsecond):
    aggregate = (np.multiply(matrixfirst, matrixsecond)).sum()
    return aggregate

#FUNCTION FOR MARKET CLEAR

def Marketclear(w):

    z = 1
    dens = 7
    kstar = ((((1 / betafirm - 1 + delta) * ((w / alpha_l) ** (alpha_l / (1 - alpha_l)))) /
             (alpha_k * (z ** (1 / (1 - alpha_l))))) **
             ((1 - alpha_l) / (alpha_k + alpha_l - 1)))
    kbar = 2*kstar
    lb_k = 0.001
    ub_k = kbar
    krat = np.log(lb_k / ub_k)
    numb = np.ceil(krat / np.log(1 - delta))
    K = np.zeros(int(numb * dens))
    for j in range(int(numb * dens)):
        K[j] = ub_k * (1 - delta) ** (j / dens)
    kgrid = K[::-1]
    sizek = kgrid.shape[0]
    
       
    ### Value function iteration
    # operating profits, op
    sizez = z_grid.shape[0]
    op = np.zeros((sizez, sizek))
    for i in range(sizez):
        for j in range(sizek):
            op[i,j] = ((1 - alpha_l) * ((alpha_l / w) ** (alpha_l / (1 - alpha_l))) *
          ((kgrid[j] ** alpha_k) ** (1 / (1 - alpha_l))) * (z_grid[i] ** (1/(1 - alpha_l))))

    # firm cash flow, e
    e = np.zeros((sizez, sizek, sizek))
    for i in range(sizez):
        for j in range(sizek):
            for k in range(sizek):
                e[i, j, k] = (op[i,j] - kgrid[k] + ((1 - delta) * kgrid[j]) -
                           ((psi / 2) * ((kgrid[k] - ((1 - delta) * kgrid[j])) ** 2)
                            / kgrid[j]))

    # Value funtion iteration
    VFtol = 1e-6
    VFdist = 7.0
    VFmaxiter = 3000
    V = np.zeros((sizez, sizek))  # initial guess at value function
    Vmat = np.zeros((sizez, sizek, sizek))  # initialize Vmat matrix
    Vstore = np.zeros((sizez, sizek, VFmaxiter))  # initialize Vstore array
    VFiter = 1

    start_time = time.clock()
    while VFdist > VFtol and VFiter < VFmaxiter:
        TV = V    
        Vmat = VFI_loop(V, e, betafirm, sizez, sizek, Vmat, pi)
        Vstore[:, :, VFiter] = V.reshape(sizez, sizek,)  # store value function at each
        # iteration for graphing later
        V = Vmat.max(axis=2)  # apply max operator to Vmat (to get V(k))
        PF = np.argmax(Vmat, axis=2)  # find the index of the optimal k'
        Vstore[:,:, i] = V  # store V at each iteration of VFI
        VFdist = (np.absolute(V - TV)).max()  # check distance between value
        # function for this iteration and value function from past iteration
        VFiter += 1
    
    VFI_time = time.clock() - start_time
    if VFiter < VFmaxiter:
        print('Value function converged after this many iterations:', VFiter)
    else:
        print('Value function did not converge')
    print('VFI took ', VFI_time, ' seconds to solve')
    
    VF = V  # solution to the functional equation


    ### Collect optimal values(functions)
    # Optimal capital stock k'
    optK = kgrid[PF]

    # optimal investment I
    optINVESTMENT = optK - (1 - delta) * kgrid

    # optimal labor demand
    optLABORDEMAND = np.zeros((sizez, sizek))
    for i in range(sizez):
        for j in range(sizek):
            optLABORDEMAND[i,j] = (((alpha_l / w) ** (1 / (1 - alpha_l))) *
          ((kgrid[j] ** alpha_k) ** (1 / (1 - alpha_l))) * (z_grid[i] ** (1/(1 - alpha_l))))


    ### Find Stationary Distribution
    Gamma = np.ones((sizez, sizek)) * (1 / (sizek * sizez))
    SDtol = 1e-12
    SDdist = 7
    SDiter = 0
    SDmaxiter = 1000
    while SDdist > SDtol and SDmaxiter > SDiter:
        HGamma = SD_loop(PF, pi, Gamma, sizez, sizek)
        SDdist = (np.absolute(HGamma - Gamma)).max()
        Gamma = HGamma
        SDiter += 1
        
    if SDiter < SDmaxiter:
        print('Stationary distribution converged after this many iterations: ',
              SDiter)
    else:
        print('Stationary distribution did not converge')
        
        
        
    #aggregate values
    # labor demand
    optAGGLABORDEMAND = aggregate(optLABORDEMAND, Gamma)

    # Investment
    optAGGINVESTMENT = aggregate(optINVESTMENT, Gamma)

    # Adjustment costs
    optADJC = psi/2 * np.multiply((optINVESTMENT)**2, 1/kgrid)
    optAGGADJC = aggregate(optADJC, Gamma)

    # Output
    optY = np.multiply(np.multiply((optLABORDEMAND) ** alpha_l, kgrid ** alpha_k),np.transpose([z_grid]))
    optAGGY = aggregate(optY, Gamma)

    # Consumption
    optCONSUMPTION = optAGGY - optAGGINVESTMENT - optAGGADJC


    ### Find labor supply
    optALS = w/(h * optCONSUMPTION)

    marketclear = abs(optALS - optAGGLABORDEMAND)
    
    # Stationary distribution in 3D -A --CONSIDER THIS ONE FIRST
    zmat, kmat = np.meshgrid(kgrid, np.log(z_grid))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_ylim(8, 12)
    ax.plot_surface(kmat, zmat, Gamma, rstride=1, cstride=1, cmap=cm.Blues,
                    linewidth=0, antialiased=False)
    ax.view_init(elev=20., azim=100)  # to rotate plot for better view
    ax.set_title(r'Fig 1. Stationary Distribution')
    ax.set_xlabel(r'Log Productivity')
    ax.set_ylabel(r'Capital Stock')
    ax.set_zlabel(r'Density')
    fig.savefig('figure1A_Stationary_Distribution.png', transparent=False, dpi=80, bbox_inches="tight")
    
     # Stationary distribution in 3D --B
    zmat, kmat = np.meshgrid(kgrid, np.log(z_grid))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_ylim(8, 12)
    ax.plot_surface(kmat, zmat, Gamma, rstride=1, cstride=1, cmap=cm.Blues,
                    linewidth=0, antialiased=False)
    ax.view_init(elev=20., azim=20)  # to rotate plot for better view
    ax.set_title(r'Fig 1. Stationary Distribution')
    ax.set_xlabel(r'Log Productivity')
    ax.set_ylabel(r'Capital Stock')
    ax.set_zlabel(r'Density')
    fig.savefig('figure1B_Stationary_Distribution.png', transparent=False, dpi=80, bbox_inches="tight")   
    
    
    
    # Plot policy function for k' -A
    zmat, kmat = np.meshgrid(kgrid, np.log(z_grid))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(kmat, zmat, optK, rstride=1, cstride=1, cmap=cm.Blues,
                    linewidth=0, antialiased=False)
    ax.view_init(elev=20., azim=100)  # to rotate plot for better view
    ax.set_title(r'Fig 2. Policy Function')
    ax.set_xlabel(r'Log Productivity')
    ax.set_ylabel(r'Capital Stock')
    ax.set_zlabel(r'Optimal Capital Stock')
    fig.savefig('figure2A_Policy_Function.png', transparent=False, dpi=80, bbox_inches="tight")
    
    
    
    
       # Plot policy function for k' --B  --CONSIDER THIS ONE FIRST
    zmat, kmat = np.meshgrid(kgrid, np.log(z_grid))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(kmat, zmat, optK, rstride=1, cstride=1, cmap=cm.Blues,
                    linewidth=0, antialiased=False)
    ax.view_init(elev=20., azim=20)  # to rotate plot for better view
    ax.set_title(r'Fig 2. Policy Function')
    ax.set_xlabel(r'Log Productivity')
    ax.set_ylabel(r'Capital Stock')
    ax.set_zlabel(r'Optimal Capital Stock')
    fig.savefig('figure2B_Policy_Function.png', transparent=False, dpi=80, bbox_inches="tight")
    
    return marketclear


# Call the minimizer
# Minimize with (truncated) Newton's method (called Newton Conjugate Gradient method)
wage_initial = w
GE_results = opt.minimize(Marketclear, wage_initial, method='Nelder-Mead', tol = 1e-12, options={'maxiter': 5000})



### WAGE RATE
opt_w = GE_results['x']
print('The equilibrium wage rate =', opt_w)

