#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss

# a = SLS                 #d73027
# b = Maxwell             #fc8d59
# c = Burgers             #fee090
# d = Extended Burgers    #91bfdb
# e = Andrade             #4575b4

################## PRELIMINARIES #####################

############ Import True Model to fit ################
J1 = np.loadtxt('../andrade_J1.dat')
J2 = np.loadtxt('../andrade_J2.dat')
w = np.loadtxt('../andrade_w.dat')
f = w/(2.*np.pi)
Tp=1./f
J = J1-1.j*J2
M = (1./J) 
Qi = J2/J1*((1.+np.sqrt(1.+(J2/J1)**2))/2.)
eta0 = 1.80242446e+21
tauM = 3.00407854e+10
y2s = 364.25*24.*60.*60
idx_seis = np.argmin(np.abs(f-1.)) # 1 Hz
E0 = np.real(M[idx_seis])

################ Fit some data points ################

def andrade(Me,B,T,w):
    n = 1./3.
    Ju = 1./Me
    J1 = Ju*(1. + B*ss.gamma(1.+n)*np.cos(n*np.pi/2.)/w**n)
    J2 = Ju*(B*ss.gamma(1.+n)*np.sin(n*np.pi/2.)/w**n + \
                 1./(w*T))
    J = J1-1.j*J2
    Qi = J2/J1*((1.+np.sqrt(1.+(J2/J1)**2))/2.)
    return J,Qi

# You have two pieces of info (Qi and Me) at two frequencies
idx_seis = np.argmin(np.abs(f-1.)) # 1 Hz
Me = np.real(M[idx_seis])
Tp_GIA = 1000*y2s
Tp_PS = 1e1*y2s

idx_GIA = np.argmin(np.abs(Tp-Tp_GIA))
Qi_GIA = Qi[idx_GIA]
idx_PS = np.argmin(np.abs(Tp-Tp_PS))
Qi_PS = Qi[idx_PS]

nB = 100
nT = 100
Bs = np.logspace(-5,-3,nB)
Ts = np.logspace(1,5,nT)*y2s
chi2 = np.zeros((nB,nT))

for ii in range(nB):
    for jj in range(nT):
        J_GIA,Qi_GIA = andrade(Me,Bs[ii],Ts[jj],w[idx_GIA])
        J_PS,Qi_PS = andrade(Me,Bs[ii],Ts[jj],w[idx_PS])
        chi2[ii,jj] = ( (np.log10(Qi_GIA)-np.log10(Qi[idx_GIA]))**2 + \
                            (np.log10(Qi_PS)-np.log10(Qi[idx_PS]))**2 )/2.

[iB,iT] = np.where(chi2==np.min(chi2))
Bfit=Bs[iB]
Tfit=Ts[iT]

# from fit, then we can extract eta_ss
eta_ss = Tfit*Me
print 'true eta_ss from fit:',eta_ss

np.savetxt('Andradefit_chi2.dat',chi2,fmt='%e')
np.savetxt('Andradefit_betas.dat',Bs,fmt='%e')
np.savetxt('Andradefit_tauMs.dat',Ts,fmt='%e')

[Jf,Qif] = andrade(Me,Bfit,Tfit,w)

# save best fit model
np.savetxt('Andradefit_Qi.dat',Qif,fmt='%e')
