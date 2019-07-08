#! /usr/bin/env python

# Equations refer to Faul & Jackson (2015)

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss
import scipy.optimize as so


# (0) SET SOME PARAMETERS
# loading parameters
tf = 100.e3      # final time (y)
dt = 50.         # time increment (y)
tmax = 75.e3     # time of maximum load (y)
tmelt = 90.e3    # time for no load left (y)
sigmax = 1.e7   # maximum load (Pa)

# rheological parameters
tm = 1000.       # Andrade Maxwell time (y)
nn = 1./3.       # Andrade exponent
Jm = 1./(60.e9)  # 1 over Maxwell modulus
bb = 1.e-4*Jm     # Andrade Beta factor

# fitting parameters
tfit = 94.e3     # time to begin fitting (y)

y2s = 364.25*24*60.*60.


# (1) DEFINE CREEP FUNCTION

def J_an(t,tm,bb,nn,Jm):
    # Andrade Creep function (eq. 6 in FJ2015)
    # t = time
    # tm = Maxwell time
    # b = Beta factor
    # n = exponent
    J_an = Jm*(1.+t/tm) + bb*t**nn
    return J_an

def J_mx(t,tm,Jm):
    # Maxwell Creep function (eq. 3 in FJ2015)
    # t = time
    # tm = Maxwell time
    J_mx = Jm*(1.+t/tm) 
    return J_mx

# (2) CREATE LOADING TIME SERIES

tau = np.arange(0,tf+dt,dt)
n = len(tau)
imax = np.where(tau==tmax)[0][0]
dsig = sigmax/np.float(imax+1.)
sig = np.zeros(n)
for i in range(1,imax+1):
    sig[i] = sig[i-1] + dsig
imelt = np.where(tau==tmelt)[0][0]
nmelt = imelt-imax
dsig = sigmax/np.float(nmelt+1.)
for i in range(imax,imelt):
    sig[i] = sig[i-1] - dsig

# (3) CONVOLVE LOAD

ep = np.zeros(n)
for i in range(1,n):
    for j in range(1,i+1):
        t = (tau[i]-tau[j])*y2s
        J_an1 = J_an(t,tm*y2s,bb,nn,Jm)
        dsig = sig[j]-sig[j-1]
        ep[i] = ep[i] + J_an1*dsig

# (4) TRY DIFFERENT 'FITS' OF MAXWELL MODELS

ifit = np.where(tau>=tfit)[0]
u_data = ep[ifit]

tm_tests = np.arange(500,2500,5)
ntest = len(tm_tests)

for itest in range(ntest):
    print itest,'of', ntest
    ep_test = np.zeros(n)
    tm_i = tm_tests[itest]
    for i in range(n):
        for j in range(1,i+1):
            t = (tau[i]-tau[j])*y2s
            J_mx1 = J_mx(t,tm_i*y2s,Jm)
            dsig = sig[j]-sig[j-1]
            ep_test[i] = ep_test[i] + J_mx1*dsig
    np.savetxt('ep_'+str(itest)+'.dat',ep_test,fmt='%e')

np.savetxt('tauM_fit.dat',tm_tests,fmt='%f')
np.savetxt('sigma_fulltime.dat',sig,fmt='%f')
np.savetxt('ep_fulltime.dat',ep,fmt='%f')
np.savetxt('fulltime.dat',tau,fmt='%f')


