#! /usr/bin/env python

# this script does the following:
#
# (1) outputs the Andrade model that we will use as the 
#     example throughout the paper
# (2) plots chi2 for the fitting figure
# (3) outputs the Andrade model from the chi2 fit that
#     will be used to demonstrate fitting.

import numpy as np
import scipy.special as ss
import matplotlib.pyplot as plt

def andrade(Me,B,T,w):
    n = 1./3.
    Ju = 1./Me
    J1 = Ju*(1. + B*ss.gamma(1.+n)*np.cos(n*np.pi/2.)/w**n)
    J2 = Ju*(B*ss.gamma(1.+n)*np.sin(n*np.pi/2.)/w**n + \
                 1./(w*T))
    J = J1-1.j*J2
    Qi = J2/J1*((1.+np.sqrt(1.+(J2/J1)**2))/2.)
    return J,Qi


################## PRELIMINARIES #####################
# frequency data
y2s = 364.25*24.*60.*60
nw = 1000
fmin = 1.e-13
fmax = 1.e1
f = np.logspace(np.log10(fmin),np.log10(fmax),nw)
w = 2.*np.pi*f
Tp = 1/f
# choose suitable unrelaxed modulus:
Gu = 60.e9 # 60 GPa

################## OBSERVED MODEL #####################
# True Model to fit
#tauM = 1.e2*y2s # 1000 years
#B = .0002
#tauM = 5.e3*y2s # 5000 years
#B = 1.e-5
tauM = 1.e3*y2s # 5000 years
B = 1.e-4
J,Qi = andrade(Gu,B,tauM,w)
M = 1./J

# save ouput for (1)
np.savetxt('andrade_J1.dat',np.real(J),fmt='%e')
np.savetxt('andrade_J2.dat',-np.imag(J),fmt='%e')
np.savetxt('andrade_w.dat',w,fmt='%e')


