#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# a = SLS                 #4daf4a
# b = Maxwell             #984ea3
# c = Burgers             #ff7f00
# d = Extended Burgers    #e41a1c
# e = Andrade             #377eb8

################## PRELIMINARIES #####################
# frequency data
w = np.loadtxt('../JF10fit/omega.dat')
f = w/(2.*np.pi)
nf=len(w)
y2s = 364.25*24.*60.*60
# choose suitable unrelaxed modulus:
Gu = 60.e9 # 60 GPa

################## OBSERVED MODEL #####################
# True Model to fit
J1 = np.loadtxt('../JF10fit/J1_eburgers.dat')
J2 = np.loadtxt('../JF10fit/J2_eburgers.dat')
J = (J1-1.j*J2) * (1./Gu)
M = (1./J) 
Qi = J2/J1*((1.+np.sqrt(1.+(J2/J1)**2))/2.)
tauM = 1.5826e+08 # run ../JF10fit/test_HL.m
E0 = np.real(M[-1])
eta0 = tauM*np.real(M[-1])
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

M0_true = np.zeros(nf,dtype=complex)
for i in range(nf):
    M0_true[i] = maxwell(E0,eta0,w[i])
v0_true = -1.j*M0_true/w


################## FITTING MODELS  #####################

fband_seis = np.array([1./50,1./1.])
fband_pseis = np.array([1./(0.25*y2s),1./((1./24)*y2s)])
fband_lake = np.array([1./(500*y2s),1./(100*y2s)])
fband_gia = np.array([1./(5000*y2s),1./(1000*y2s)])
f_gia=10.**(np.mean(np.log10(fband_gia)))
eta_gia = 1.3424417218e+19
f_lake=10.**(np.mean(np.log10(fband_lake)))
eta_lake = 1.30392324099e+19
f_pseis=10.**(np.mean(np.log10(fband_pseis)))
eta_mx_pseis = 4.6304679387e+17
eta_b1_pseis = 6.84564762544e+18
eta_b2_pseis = 1.05950441708e+17
f_seis=10.**(np.mean(np.log10(fband_seis)))
eta_seis = 1.52385728732e+14
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



# reference model (using inferred eta from GIA)
M0 = np.zeros(nf,dtype=complex)
for i in range(nf):
    M0[i] = maxwell(E0,eta_gia,w[i])
v0 = -1.j*M0/w



plt.figure(1,figsize=(7,9))

plt.subplot(3,1,1)


plt.loglog([1e100,1e100],[100,100],'o-',lw=3,color='#4daf4a',mew=0,ms=10,mfc='#4daf4a')
plt.loglog([1e100,1e100],[100,100],'o-',lw=3,color='#984ea3',mew=0,ms=10,mfc='#984ea3')
plt.loglog([1e100,1e100],[100,100],'o-',lw=3,color='#ff7f00',mew=0,ms=10,mfc='#ff7f00')
plt.legend(['fit with Zener','fit with Maxwell','fit with Burgers'],\
           loc='lower left',numpoints=1,frameon=False)



plt.loglog([f[0],f[-1]],[eta0,eta0],'k-',lw=4)


plt.loglog([f_gia,f_gia],[eta_gia,eta_gia],'o',mew=2,color='#984ea3',ms=12)
plt.loglog([f_lake,f_lake],[eta_lake,eta_lake],'o',mew=2,color='#984ea3',ms=12)
plt.loglog([f_pseis,f_pseis],[eta_mx_pseis,eta_mx_pseis],'o',mew=2,color='#984ea3',ms=12)
plt.loglog([f_pseis,f_pseis],[eta_b1_pseis,eta_b1_pseis],'o',mew=2,color='#ff7f00',ms=12)
plt.loglog([f_pseis,f_pseis],[eta_b2_pseis,eta_b2_pseis],'o',mew=2,color='#ff7f00',ms=12)
plt.loglog([f_seis,f_seis],[eta_seis,eta_seis],'o',mew=2,color='#4daf4a',ms=12)

plt.ylabel(r'$\eta$',fontsize=14)
plt.ylim([1.e13,1.e20])
plt.yticks([1e14,1e16,1e18,1e20])
plt.xticks([1e-12,1e-9,1e-6,1e-3,1e0])
plt.xlim([1e-12,1.e0])
plt.xlabel(r'$f$ (Hz)',fontsize=14)

plt.subplot(3,1,2)

plt.semilogx(f,np.abs(v)/np.abs(v0),'k-',lw=4)
plt.semilogx(f,np.abs(v)/np.abs(v0_true),'k--',lw=4)

plt.legend([r'$\bar{\eta}^*_{\rm true}$',r'$\bar{\eta}^*_{\rm rel}$'],\
           loc='lower right',frameon=False)

idx=np.where( (f>fband_seis[0]) & (f<fband_seis[1]) )[0]
fac = np.abs(v_seis)/np.abs(v0)
plt.semilogx(f,fac,color='#4daf4a',lw=3,alpha=0.3)
plt.semilogx(f[idx],fac[idx],color='#4daf4a',lw=6,zorder=10)

idx=np.where( (f>fband_pseis[0]) & (f<fband_pseis[1]) )[0]
fac = np.abs(v_mx_pseis)/np.abs(v0)
plt.semilogx(f,fac,color='#984ea3',lw=3,alpha=0.3)
plt.semilogx(f[idx],fac[idx],color='#984ea3',lw=6,zorder=10)

fac = np.abs(v_bg_pseis)/np.abs(v0)
plt.semilogx(f,fac,color='#ff7f00',lw=3,alpha=0.3)
plt.semilogx(f[idx],fac[idx],color='#ff7f00',lw=6,zorder=10)

idx=np.where( (f>fband_lake[0]) & (f<fband_lake[1]) )[0]
fac = np.abs(v_lake)/np.abs(v0)
plt.semilogx(f,fac,color='#984ea3',lw=3,alpha=0.3)
plt.semilogx(f[idx],fac[idx],color='#984ea3',lw=6,zorder=10)

idx=np.where( (f>fband_gia[0]) & (f<fband_gia[1]) )[0]
fac = np.abs(v_gia)/np.abs(v0)
plt.semilogx(f,fac,color='#984ea3',lw=3,alpha=0.3)
plt.semilogx(f[idx],fac[idx],color='#984ea3',lw=6,zorder=10)

plt.ylabel(r'$\bar{\eta}^*$',fontsize=14)
plt.ylim([0,2.0])
plt.yticks([0.5,1.0,1.5,2.0])
plt.xticks([1e-12,1e-9,1e-6,1e-3,1e0])
plt.xlim([1e-12,1.e0])
plt.xlabel(r'$f$ (Hz)',fontsize=14)


plt.subplot(3,1,3)

Q1=1./(f*1e+18)
Q2=1./(f*1e+12)
Q3=1./(f*1e+6)
Q5=1./(f*1e+3)

Q4=1./(f*1.1e+9)


plt.loglog(f,Qi,'k-',lw=4)
plt.loglog(f,Q1,'k:',lw=1)
plt.loglog(f,Q2,'k:',lw=1)
plt.loglog(f,Q3,'k:',lw=1)
plt.loglog(f,Q4,'k:',lw=1)
plt.loglog(f,Q5,'k:',lw=1)

plt.ylim([1e-6,1e6])
plt.yticks([1e-3,1e0,1e3,1e6])
plt.ylabel(r'$Q^{-1}$',fontsize=14)
plt.legend([r'$Q^{-1}_{\rm true}$'],\
           loc='upper right',frameon=False)


plt.xticks([1e-12,1e-9,1e-6,1e-3,1e0])
plt.xlim([1e-12,1.e0])
plt.xlabel(r'$f$ (Hz)',fontsize=14)


plt.tight_layout()

plt.savefig('fit.pdf')

plt.show()
