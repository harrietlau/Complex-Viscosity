#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pp

# a = SLS                 #d73027
# b = Maxwell             #fc8d59
# c = Burgers             #fee090
# d = Extended Burgers    #91bfdb
# e = Andrade             #4575b4

y2s = 364.25*24.*60.*60


# TRUE MODEL
J1 = np.loadtxt('../andrade_J1.dat')
J2 = np.loadtxt('../andrade_J2.dat')
w = np.loadtxt('../andrade_w.dat')
f = w/(2.*np.pi)
J = J1-1.j*J2
M = (1./J) 
Qi = J2/J1*((1.+np.sqrt(1.+(J2/J1)**2))/2.)
eta0= 1.97810002e+21
tauM = 1000.*y2s


# FITS
# (1) from running step1_calcfits.py
# frequencies
f_gia=1./(1000*y2s) # 5000 y
f_lake=1./(250*y2s) # 250 y
f_pseis=1./(y2s/(52.)) # ~ week
f_seis=1./1.
# fit viscosities
eta_gia=7.08669258119e+20
eta_lake=2.50755147372e+20
eta_mx_pseis=9.68237366664e+16
eta_b1_pseis=2.24476338e+13
eta_b2_pseis=3.78961698e+19
eta_seis=3.34078009032e+15

# PLOT FIG 3A
fig=plt.figure(1,figsize=(7,3.2))
plt.loglog([1.e-13,1.e1],[eta0,eta0],'k-',lw=2)
plt.loglog([f_gia,f_gia],[eta_gia,eta_gia],'o',\
               mew=1,color='#fc8d59',ms=12)
plt.loglog([f_lake,f_lake],[eta_lake,eta_lake],'o',\
               mew=1,color='#fc8d59',ms=12)
plt.loglog([f_pseis,f_pseis],[eta_mx_pseis,eta_mx_pseis],\
               'o',mew=1,color='#fc8d59',ms=12)
plt.loglog([f_pseis,f_pseis],[eta_b1_pseis,eta_b1_pseis],\
               's',mew=1,color='#fee090',ms=12)
plt.loglog([f_pseis,f_pseis],[eta_b2_pseis,eta_b2_pseis],\
               'o',mew=1,color='#fee090',ms=12)
plt.loglog([f_seis,f_seis],[eta_seis,eta_seis],'s',\
               mew=1,color='#d73027',ms=12)

plt.xlim([1e-13,1e1])
plt.ylim([1e12,1e24])
plt.xlabel(r'$f$ (Hz)',fontsize=14)
plt.ylabel(r'$\eta$ (Pa s)',fontsize=14)
plt.xticks([1e-13,1e-11,1e-9,1e-7,1e-5,1e-3,1e-1,1e1])
plt.yticks([1e14,1e16,1e18,1e20,1e22,1e24])
plt.tight_layout()
plt.savefig('fig3a.pdf')



# PLOT FIG 3B

Qif=np.loadtxt('Andradefit_Qi.dat')

# find some lines
i_seis = np.argmin(np.abs(f-1.)) # 1 Hz
i_gia = np.argmin(np.abs(f-f_gia)) # 1000 y

fig=plt.figure(2,figsize=(7,3.2))
plt.loglog(f,(1./f)*4.e1,'k:')
plt.loglog(f,(1./f)*4.e-2,'k:')
plt.loglog(f,(1./f)*4e-5,'k:')
plt.loglog(f,(1./f)*.5e-8,'k:')
plt.loglog(f,(1./f)*.6e-11,'k:')
plt.loglog(f,(1./f)*.6e-14,'k:')
plt.loglog(f,(1./f)*.6e-17,'k:')

plt.loglog(f,Qi,'k-',lw=2)
plt.loglog([f[i_seis],f[i_seis]],[Qi[i_seis],Qi[i_seis]],\
               '^',mew=1,mec='k',mfc='c',ms=12)
plt.loglog([f[i_gia],f[i_gia]],[Qi[i_gia],Qi[i_gia]],\
               'o',mew=1,mec='k',mfc='c',ms=12)
itrim=range(0,len(f),50)
plt.loglog(f[itrim],Qif[itrim],'x',mec='green',mew=2,ms=5,zorder=0)

plt.xlim([1e-13,1e1])
plt.ylim([1e-12,1e6])
plt.xlabel(r'$f$ (Hz)',fontsize=14)
plt.ylabel(r'$Q^{-1}$',fontsize=14)
plt.xticks([1e-13,1e-11,1e-9,1e-7,1e-5,1e-3,1e-1,1e1])
plt.yticks([1e-9,1e-6,1e-3,1e0,1e3,1e6])
plt.tight_layout()
plt.savefig('fig3b.pdf')

fig=plt.figure(3,figsize=(4,3))

chi2=np.loadtxt('Andradefit_chi2.dat')
Bs=np.loadtxt('Andradefit_betas.dat')
Ts=np.loadtxt('Andradefit_tauMs.dat')
print Bs[-1]
[xx,yy]=np.meshgrid(np.log10(Ts/y2s),np.log10(Bs))
tauM=1.e3*y2s # True value from ../calc_model.py
B=1.e-4 # True value from ../calc_model.py

cmax=0.9
[r,c]=np.where(chi2>=cmax)
chi2[r,c]=cmax

ax=plt.gca()
ccontours=np.array([0.,0.2,0.4,0.6,0.8,1.0])
plt.contourf(xx,yy,chi2,ccontours,cmap='OrRd')
plt.xticks([1.0,3.0,5.0],\
               [r'10$^1$',r'10$^3$',r'10$^5$'],\
               fontsize=14)
plt.yticks([-5.,-4.0,-3.0],\
               [r'10$^{-5}$',r'10$^{-4}$',r'10$^{-3}$'],\
               fontsize=14)
cbar=plt.colorbar()
cbar.ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0])
cbar.set_label(r'$\chi^2$')
plt.xlabel(r'$\tau_M$',fontsize=14)
plt.ylabel(r'$\beta$',fontsize=14)
plt.plot([np.log10(tauM/y2s),np.log10(tauM/y2s)],[np.log10(B),np.log10(B)],\
             'o',mfc='g',ms=10,mew=0)

plt.tight_layout()

plt.savefig('fig3c.pdf')

[iB,iT] = np.where(chi2==np.min(chi2))
Bfit=Bs[iB]
Tfit=Ts[iT]
print "best fit beta:", Bs[iB]
print "best fit TauM:", Ts[iT]

plt.show()
