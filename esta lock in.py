# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 22:42:47 2021

@author: Maia
"""

import os
import numpy as np
import matplotlib.pyplot as plt
os.chdir (r'C:\Users\54116\Desktop\Labo 4\lock in II\Largo total')
data = np.loadtxt("5k.40k.txt", delimiter = " ",skiprows=1)
#%%

frec=data[:,0]
r=data[:,1]
theta=data[:,2]
x=data[:,3]
y=data[:,4]
#%%
plt.plot(frec,y)
#%%
from scipy.optimize import curve_fit 
vo= 0.5
R= 1000
def f(w, L, Ra):
    return vo*w*L*R/((Ra+R)**2+w**2*L**2)
po= [100, 0.01]
param, param_cov = curve_fit(f, frec, y, p0=po) 
print(param)
#%%
plt.plot(frec,y)
x=np.linspace(5000,40000,100000)
plt.plot(x, [f(i,param[0], param[1]) for i in x])
#%%
from scipy.optimize import curve_fit 
vo= 0.5
R= 1000
def f(w, L, Ra):
    return (vo/((Ra+R)**2+w**2*L**2))*np.sqrt((Ra*(R+Ra)+w**2*L**2)**2+w**2*L**2*R**2)
po= [100, 0.01]
param, param_cov = curve_fit(f, frec, r, p0=po) 
print(param)
