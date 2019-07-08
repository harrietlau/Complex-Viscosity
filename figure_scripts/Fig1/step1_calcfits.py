#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

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
J = J1-1.j*J2
M = (1./J) 
Qi = J2/J1*((1.+np.sqrt(1.+(J2/J1)**2))/2.)
eta0 = 1.80242446e+21
tauM = 3.00407854e+10
y2s = 364.25*24.*60.*60
idx_seis = np.argmin(np.abs(f-1.)) # 1 Hz
E0 = np.real(M[idx_seis])

##################### GIA #####################
# Use Maxwell 
f_gia=1./(1000*y2s) # 5000 y
i = np.abs(f_gia-f).argmin()
# solve for E and eta
E = E0
# solve using Qi:
tau = 1./(Qi[i]*w[i])
eta = tau*E
# solve using M:
eta = np.sqrt( np.real(M[i])/(w[i]**2*(1./E - np.real(M[i])/(E*E))) )

print 'For GIA band:','E = ',E,'; Eta = ',eta

################ Lake Rebound ##################
# Use Maxwell 
f_lake=1./(250*y2s) # 250 y
i = np.abs(f_lake-f).argmin()
# solve for E and eta
E = E0
# solve using Qi:
tau = 1./(Qi[i]*w[i])
eta = tau*E
# solve using M:
eta = np.sqrt( np.real(M[i])/(w[i]**2*(1./E - np.real(M[i])/(E*E))) )

print 'For Lake rebound band:','E = ',E,'; Eta = ',eta

################ Post Seismic ##################

# Use Maxwell 
f_pseis=1./(y2s/(52.)) # ~ week
i = np.abs(f_pseis-f).argmin()
# solve for E and eta
E = E0
# solve using Qi:
tau = 1./(Qi[i]*w[i])
eta = tau*E
## solve using M:
eta = np.sqrt( np.real(M[i])/(w[i]**2*(1./E - np.real(M[i])/(E*E))) )

print 'For Post Seismic Relaxation (Maxwell) band:','E = ',E,'; Eta = ',eta

# Use Burgers 
E1 = E0
E2 = E0
Mr = np.real(M[i])
Qii = Qi[i]

ntest=300
# refined after bigger search
eta1s = np.logspace(17,20,ntest) 
eta2s = np.logspace(12,14,ntest)

def burgers(E1,E2,eta1,eta2,w):
    M=(1.j*w*eta1*(1.+1.j*w*eta2/E2)) / \
        (1.+1.j*w*((eta1/E2 + eta1/E1 + eta2/E2) + 1.j*w*eta1*eta2/(E1*E2)))
    Qi = np.imag(M)/np.real(M)
    return M,Qi

chi2 = np.zeros((ntest,ntest))

for ii in range(ntest):
    for jj in range(ntest):
        M_B,Qi_B = burgers(E1,E2,eta1s[ii],eta2s[jj],w[i])
        chi2[ii,jj] = np.abs(Qi_B-Qii)

[r,c]=np.where(chi2==np.min(chi2))
eta1 = eta1s[r]
eta2 = eta2s[c]

print 'For postseismic rebound band (Burgers):',\
    'E1 = ',E1,'E2 = ',E2,'; Eta1 = ',eta1,'; Eta2 = ',eta2

################### Seismic ###################
f_seis=1./25.
i = np.abs(f_seis-f).argmin()

E1=E0
E2=E0
wr=w[i]

eta = E1*(wr+np.sqrt(wr*wr-8.*wr*wr*Qi[i]*Qi[i]))/(2.*wr*wr*Qi[i])
print 'For seismic band (Zener):',\
    'E1 = ',E1,'E2 = ',E2,'; Eta = ',eta
