# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 22:39:24 2021

@author: Maia
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
os.chdir (r'C:\Users\54116\Desktop\Labo 4\lock in II\27-10\Con vueltas')
data = np.loadtxt("5000_40000promVsstd.txt", delimiter = " ",skiprows=1)
#%%

frec=data[:,0]
r=data[:,1]
rstd=data[:,2]
theta=data[:,3]
thetastd=data[:,4]
x=data[:,5]
xstd=data[:,6]
y=data[:,7]
ystd=data[:,8]
#%%
eyi=0.002*y
eys=ystd/np.sqrt(20)
errory=np.sqrt(eyi**2+eys**2)

exi=0.002*x
exs=xstd/np.sqrt(20)
errorx=np.sqrt(exi**2+exs**2)

X=y/x

eX=np.sqrt((errory/x)**2+(errorx*y/x**2)**2)

plt.errorbar(frec,X,yerr=eX, linestyle='none')

#%%
plt.plot(frec,np.arctan(theta))

#%%
X=y/x
def f1(w,C):
    return C*w
param, pcov = curve_fit(f1, frec, X, sigma=eX) 
print(param)
print(np.sqrt(np.diag(pcov)))
l=np.linspace(5000,40000,100000)
plt.plot(l, [f1(i,param[0]) for i in l])
plt.errorbar(frec,X,yerr=eX, marker='.', linestyle='none')
#%%
def f2(w,k):
    return k*w
param, pcov = curve_fit(f2, frec, y, sigma=errory) 
print(param)
print(np.sqrt(np.diag(pcov)))
#%%
erry=np.mean(errory)
errX=np.mean(eX)
print(erry)
#%%
os.chdir (r'C:\Users\54116\Desktop\Labo 4\lock in II\Largo 2')
data = np.loadtxt("5k.40k.txt", delimiter = " ",skiprows=1)
#%%

frec=data[:,0]
r=data[:,1]
theta=data[:,2]
x=data[:,3]
y=data[:,4]
#%%
eyi=0.002*y
eys=erry
errory=np.sqrt(eyi**2+eys**2)

exi=0.002*x
exs=errX
errorx=np.sqrt(exi**2+exs**2)

X=y/x

eX=np.sqrt((errory/x)**2+(errorx*y/x**2)**2)

plt.errorbar(frec,X,yerr=eX, linestyle='none')

#%%
plt.plot(frec,y)

#%%
X=y/x
def f1(w,C):
    return C*w
param, pcov = curve_fit(f1, frec, X) 
print(param)
print(np.sqrt(np.diag(pcov)))
l=np.linspace(5000,40000,100000)
plt.plot(l, [f1(i,param[0]) for i in l])
plt.plot(frec,X, marker='.', linestyle='none')
#%%
def f2(w,k):
    return k*w
param, pcov = curve_fit(f2, frec, y) 
print(param)
print(np.sqrt(np.diag(pcov)))
#%%
erry=np.mean(errory)
errX=np.mean(eX)
