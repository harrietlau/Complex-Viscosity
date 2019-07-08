#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss

# a = SLS                 #d73027
# b = Maxwell             #fc8d59
# c = Burgers             #fee090
# d = Andrade             #4575b4

################## PRELIMINARIES #####################
# frequency data
y2s = 364.25*24.*60.*60
w = np.loadtxt('../andrade_w.dat')
f = w/(2.*np.pi)
nf= len(f)
################## OBSERVED MODEL #####################
# True Model to fit
J1 = np.loadtxt('../andrade_J1.dat')
J2 = np.loadtxt('../andrade_J2.dat')
J = J1-1.j*J2
M = (1./J) 
Qi = J2/J1*((1.+np.sqrt(1.+(J2/J1)**2))/2.)
eta0=1.80242446e+21
tauM = 3.00407854e+10
E0 = np.real(M[-1])
v = -1.j*M/w

# functions to fit
def sls(E1,E2,eta,w):
    M=(E1*E2/(E1+E2)) * ( (1+1.j*w*eta/E2)/(1.+1.j*w*(eta/(E1+E2))) )
    return M
def maxwell(E,eta,w):
    M=1.j*w*eta/(1.+1.j*w*eta/E)
    return M
def burgers(E1,E2,eta1,eta2,w):
    M=(1.j*w*eta1*(1.+1.j*w*eta2/E2)) / \
        (1.+1.j*w*((eta1/E2 + eta1/E1 + eta2/E2) + 1.j*w*eta1*eta2/(E1*E2)))
    return M

Mref=np.zeros(nf,dtype=complex)
for i in range(nf):
    Mref[i] = maxwell(E0,eta0,w[i])
vref = -1.j*Mref/w

################## FITTING MODELS  #####################

fband_seis = np.array([0.5,1.5])
fband_pseis = np.array([1./((1.5/52)*y2s),1./((0.5/52.)*y2s)])
fband_lake = np.array([1./(500*y2s),1./(300*y2s)])
fband_gia = np.array([1./(1800*y2s),1./(1200*y2s)])

eta_gia=7.08669258119e+20
eta_lake=2.50755147372e+20
eta_mx_pseis=9.68237366664e+16
eta_b1_pseis=3.78961698e+19
eta_b2_pseis=2.24476338e+13
eta_seis=3.34078009032e+15

# GIA
Ms_gia = np.zeros(nf,dtype=complex)
for i in range(nf):
    Ms_gia[i]=maxwell(E0,eta_gia,w[i])
v_gia = -1.j*Ms_gia/w
# Lake rebound
Ms_lake = np.zeros(nf,dtype=complex)
for i in range(nf):
    Ms_lake[i]=maxwell(E0,eta_lake,w[i])
v_lake = -1.j*Ms_lake/w
# Post Seismic
Ms_mx_pseis = np.zeros(nf,dtype=complex)
for i in range(nf):
    Ms_mx_pseis[i]=maxwell(E0,eta_mx_pseis,w[i])
v_mx_pseis = -1.j*Ms_mx_pseis/w
Ms_bg_pseis = np.zeros(nf,dtype=complex)
for i in range(nf):
    Ms_bg_pseis[i]=burgers(E0,E0,eta_b1_pseis,\
                           eta_b2_pseis,w[i])
v_bg_pseis = -1.j*Ms_bg_pseis/w
# Seismic
Ms_seis = np.zeros(nf,dtype=complex)
for i in range(nf):
    Ms_seis[i]=sls(E0,E0,eta_seis,w[i])
v_seis = -1.j*Ms_seis/w

# a = SLS                 #d73027
# b = Maxwell             #fc8d59
# c = Burgers             #fee090
# d = Andrade             #4575b4


plt.figure(1,figsize=(7,3.2))
plt.semilogx(f,np.abs(v)/np.abs(vref),'k-',lw=4)

plt.legend([r'$\bar{\eta}^*_{\rm mod}$'],\
           loc='lower right',frameon=False)
idx=np.where( (f>fband_seis[0]) & (f<fband_seis[1]) )[0]
fac = np.abs(v_seis)/np.abs(vref)
plt.semilogx(f,fac,color='#d73027',lw=3,alpha=0.3)
plt.semilogx(f[idx],fac[idx],color='#d73027',lw=6,zorder=10)

idx=np.where( (f>fband_pseis[0]) & (f<fband_pseis[1]) )[0]
fac = np.abs(v_mx_pseis)/np.abs(vref)
plt.semilogx(f,fac,color='#fc8d59',lw=3,alpha=0.3)
plt.semilogx(f[idx],fac[idx],color='#fc8d59',lw=6,zorder=10)

#fac = np.abs(v_bg_pseis)/np.abs(vref)
#plt.semilogx(f,fac,color='#fee090',lw=3,alpha=0.3)
#plt.semilogx(f[idx],fac[idx],color='#fee090',lw=6,zorder=10)

idx=np.where( (f>fband_lake[0]) & (f<fband_lake[1]) )[0]
fac = np.abs(v_lake)/np.abs(vref)
plt.semilogx(f,fac,color='#fc8d59',lw=3,alpha=0.3)
plt.semilogx(f[idx],fac[idx],color='#fc8d59',lw=6,zorder=10)

idx=np.where( (f>fband_gia[0]) & (f<fband_gia[1]) )[0]
fac = np.abs(v_gia)/np.abs(vref)
plt.semilogx(f,fac,color='#fc8d59',lw=3,alpha=0.3)
plt.semilogx(f[idx],fac[idx],color='#fc8d59',lw=6,zorder=10)

plt.xlim([1e-13,1.e1])
plt.ylim([0.5,1.1])
plt.xlabel(r'$f$ (Hz)',fontsize=14)
plt.ylabel(r'$\bar{\eta}^*$',fontsize=14)
plt.xticks([1e-13,1e-11,1e-9,1e-7,1e-5,1e-3,1e-1,1e1])
plt.yticks([0.6,0.7,0.8,0.9,1.0,1.1])
plt.tight_layout()

plt.savefig('fig4d.pdf')
plt.show()

