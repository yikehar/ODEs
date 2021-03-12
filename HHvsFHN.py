import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import odeint

#Constants
C_m = 1.0       #Membrane capacitance, in uF/cm**2
g_Na = 120.0    #Maximum sodium conductance, in mS/cm**2
g_K = 36.0      #Maximum potassium conductance, in mS/cm**2
g_L = 0.3       #Maximum leak conductance, in mS/cm**2
E_Na = 50.0     #Sodium Nernst reversal potential, in mV
E_K = -77.0     #Potassium Nernst reversal potential, in mV
E_L = -54.387   #Lreak Nernst reversal potentials, in mV

#Channel gating kinetics. Functions of membrane voltage
def alpha_m(V):
    return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

def beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)

def alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0) / 20.0)

def beta_h(V):
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

def alpha_n(V):
    return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

def beta_n(V):
    return 0.125 * np.exp(-(V + 65.0) / 80.0)

#Membrane currents
#Sodium
def I_Na(V, m, h):
    return g_Na * m**3 * h * (V - E_Na)

#Potassium
def I_K(V, n):
    return g_K * n**4 * (V - E_K)

#Leak
def I_L(V):
    return g_L * (V - E_L)

#Injected current
def I_inj(t, timecount):
    y = np.sin((t + timecount*dt*timestep)/5)
    I = 30 * np.where(y > 0, 1, 0)
    return I

def dALLdt(X, t):
    global timecount
    V, m, h, n = X

    dVdt = (I_inj(t, timecount) - I_Na(V, m, h) - I_K(V, n) - I_L(V)) / C_m
    dmdt = alpha_m(V) * (1.0 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1.0 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1.0 - n) - beta_n(V) * n
    return dVdt, dmdt, dhdt, dndt

