#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss

# a = SLS                 #d73027
# b = Maxwell             #fc8d59
# c = Burgers             #fee090
# d = Andrade             #4575b4

# TRUE MODEL
J1d = np.loadtxt('../andrade_J1.dat')
J2d = np.loadtxt('../andrade_J2.dat')
w = np.loadtxt('../andrade_w.dat')
f = w/(2.*np.pi)
Jd = J1d-1.j*J2d
Md = (1./Jd) 
Qid = J2d/J1d*((1.+np.sqrt(1.+(J2d/J1d)**2))/2.)
eta0=1.80242446e+21
tauM = 3.00407854e+10
E0 = np.real(Md[-1])


## SLS
E1=E0
E2=E0*0.8
eta1=eta0
eta2=eta0*1.e-1
Ma = (E1*E2/(E1+E2)) * ( (1+1.j*w*eta2/E2)/(1.+1.j*w*(eta2/(E1+E2))) )
M1 = np.real(Ma)
M2 = np.imag(Ma)
Qia = M2/M1*((1.+np.sqrt(1.+(M2/M1)**2))*0.5 )**(-1)

## MAXWELL
Mb = 1.j*w*eta1/(1.+1.j*w*eta1/E1)
M1 = np.real(Mb)
M2 = np.imag(Mb)
Qib = M2/M1*((1.+np.sqrt(1.+(M2/M1)**2))*0.5 )**(-1)
Qib_approx = M2/M1


## BURGERS
Mc = (1.j*w*eta1*(1.+1.j*w*eta2/E2)) / \
     (1.+1.j*w*((eta1/E2 + eta1/E1 + eta2/E2) + 1.j*w*eta1*eta2/(E1*E2)))
M1 = np.real(Mc)
M2 = np.imag(Mc)
Qic = M2/M1*((1.+np.sqrt(1.+(M2/M1)**2))*0.5 )**(-1)

va = -1.j*Ma/w
vb = -1.j*Mb/w
vc = -1.j*Mc/w
vd = -1.j*Md/w


# PLOT FIG 4A
# a = SLS                 #d73027
# b = Maxwell             #fc8d59
# c = Burgers             #fee090
# d = Andrade             #4575b4

plt.figure(1,figsize=(7,3.2))
plt.loglog(f,np.abs(va),'-',lw=2,color='#d73027')
plt.loglog(f,np.abs(vb),'-',lw=2,color='#fc8d59')
plt.loglog(f,np.abs(vc),'-',lw=2,color='#fee090')
plt.loglog(f,np.abs(vd),'-',lw=2,color='#4575b4')
plt.legend(['Zener','Maxwell','Burgers','Andrade'],\
               loc='upper right',frameon=False)
plt.loglog([1./tauM,1./tauM],[1e12,1e24],'k--')

plt.xlim([1e-13,1.e1])
plt.ylim([1e12,1e24])
plt.xlabel(r'$f$ (Hz)',fontsize=14)
plt.ylabel(r'$\|\eta^*\|$ (Pa s)',fontsize=14)
plt.xticks([1e-13,1e-11,1e-9,1e-7,1e-5,1e-3,1e-1,1e1])
plt.yticks([1e14,1e16,1e18,1e20,1e22,1e24])
plt.tight_layout()
plt.savefig('fig4a.pdf')


plt.figure(2,figsize=(3,1.8))
plt.loglog(f,np.abs(va),'-',lw=2,color='#d73027')
plt.loglog(f,np.abs(vb),'-',lw=2,color='#fc8d59')
plt.loglog(f,np.abs(vc),'-',lw=2,color='#fee090')
plt.loglog(f,np.abs(vd),'-',lw=2,color='#4575b4')
plt.loglog([1./tauM,1./tauM],[1e20,1e22],'k--')

plt.xlim([1e-12,1.e-9])
plt.ylim([1e20,1e22])
plt.xticks([1e-12,1e-11,1e-10,1e-9])
plt.yticks([1e21,1e22])
plt.tight_layout()
plt.savefig('fig4a_inset.pdf')


fig=plt.figure(3,figsize=(7,3.2))
plt.loglog(f,Qia,'-',lw=2,color='#d73027')
plt.loglog(f,Qib,'-',lw=2,color='#fc8d59')
plt.loglog(f,Qib_approx,'--',lw=2,color='#fc8d59')
plt.loglog(f,Qic,'-',lw=2,color='#fee090')
plt.loglog(f,Qid,'-',lw=2,color='#4575b4')
plt.loglog([1./tauM,1./tauM],[1e-8,1e4],'k--')

plt.xlim([1e-13,1.e1])
plt.ylim([1e-8,1e4])
plt.xlabel(r'$f$ (Hz)',fontsize=14)
plt.ylabel(r'$Q^{-1}$',fontsize=14)
plt.xticks([1e-13,1e-11,1e-9,1e-7,1e-5,1e-3,1e-1,1e1])
plt.yticks([1e-6,1e-4,1e-2,1e0,1e2,1e4])
plt.tight_layout()

plt.savefig('fig4b.pdf')


plt.figure(4,figsize=(3,1.8))
plt.loglog(f,Qia,'-',lw=2,color='#d73027')
plt.loglog(f,Qib,'-',lw=2,color='#fc8d59')
plt.loglog(f,Qib_approx,'--',lw=2,color='#fc8d59')
plt.loglog(f,Qic,'-',lw=2,color='#fee090')
plt.loglog(f,Qid,'-',lw=2,color='#4575b4')
plt.loglog([1./tauM,1./tauM],[1e-2,1e2],'k--')

plt.xlim([1e-12,1.e-9])
plt.ylim([1e-2,1e2])
plt.xticks([1e-12,1e-11,1e-10,1e-9])
plt.yticks([1e0,1e2])
plt.tight_layout()
plt.savefig('fig4b_inset.pdf')




fig=plt.figure(5,figsize=(7,3.2))

plt.semilogx(f,np.abs(va)/np.abs(vb),'-',lw=2,color='#d73027')
plt.semilogx(f,np.abs(vb)/np.abs(vb),'-',lw=2,color='#fc8d59')
plt.semilogx(f,np.abs(vc)/np.abs(vb),'-',lw=2,color='#fee090')
plt.semilogx(f,np.abs(vd)/np.abs(vb),'-',lw=2,color='#4575b4')

plt.semilogx([1./tauM,1./tauM],[0,1.5],'k--')

plt.xlim([1e-13,1.e1])
plt.ylim([0,1.5])
plt.xlabel(r'$f$ (Hz)',fontsize=14)
plt.ylabel(r'$\bar{\eta}^*$',fontsize=14)
plt.xticks([1e-13,1e-11,1e-9,1e-7,1e-5,1e-3,1e-1,1e1])
plt.yticks([0.25,0.5,0.75,1.0,1.25,1.5])
plt.tight_layout()
plt.savefig('fig4c.pdf')


plt.figure(6,figsize=(3,1.8))

plt.semilogx(f,np.abs(va)/np.abs(vb),'-',lw=2,color='#d73027')
plt.semilogx(f,np.abs(vb)/np.abs(vb),'-',lw=2,color='#fc8d59')
plt.semilogx(f,np.abs(vc)/np.abs(vb),'-',lw=2,color='#fee090')
plt.semilogx(f,np.abs(vd)/np.abs(vb),'-',lw=2,color='#4575b4')
plt.semilogx([1./tauM,1./tauM],[0.4,1.2],'k--')

plt.xlim([1e-12,1.e-9])
plt.ylim([0.4,1.2])
plt.xticks([1e-12,1e-11,1e-10,1e-9])
plt.yticks([0.8,1.2])
plt.tight_layout()
plt.savefig('fig4c_inset.pdf')



## MAXWELL - TEST
w=np.logspace(-25,0,100)
f = w/(2.*np.pi)
Mb = 1.j*w*eta1/(1.+1.j*w*eta1/E1)
M1 = np.real(Mb)
M2 = np.imag(Mb)
Qib = M2/M1*((1.+np.sqrt(1.+(M2/M1)**2))*0.5 )**(-1)
Qib_approx = M2/M1


plt.figure(7,figsize=(7,3.2))
plt.loglog(f,Qib,'-',lw=2,color='#fc8d59')
plt.loglog(f,Qib_approx,'--',lw=2,color='#fc8d59')
plt.loglog([1./tauM,1./tauM],[1e-2,1e2],'k--')

#plt.xlim([1e-12,1.e-9])
#plt.ylim([1e-2,1e2])
#plt.xticks([1e-12,1e-11,1e-10,1e-9])
#plt.yticks([1e0,1e2])
plt.tight_layout()


plt.show()
