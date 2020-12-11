# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:23:34 2020

@author: alexx
"""

"""
parametrii functiei odeint 
functia f 
condiditia initiala 
si timp + interval 
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.integrate import odeint
from numpy import linspace, zeros, exp

e = 2.71828

#Euler function 
def forwardEuler(f, U_0, dt, T):
    N_t = int(round(float(T)/dt))
    u = zeros(N_t+1)
    t = linspace(0, N_t*dt, len(u))
    u[0] = U_0
    for n in range(N_t):
        u[n+1] = u[n] + dt*f(u[n], t[n])
    return u, t

#function that return dy/dt
def g(y, t):
    k = 0.3
    dydt = -k * y
    return dydt

def ydx(x):
    fr = np.log(x*x - 1, e)

#initial condition for f 
y0 = 5

#conditia initiala 
y2 = 1/np.log(3)

tracef = np.linspace(0, 20)

def populationGrowthDemo():
    """Test case: u'=r*u, u(0)=100."""
    def f(u, t):
        return 0.1*u

    u, t = forwardEuler(f=f, U_0=100, dt=0.5, T=20)
    plt.plot(t, u, t, 100*exp(0.1*t))
    plt.show()

if __name__ == '__main__':
    populationGrowthDemo()
    
    