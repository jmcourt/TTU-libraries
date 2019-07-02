import numpy as np

# ======== Some useful functions for fitting (i.e. to power spectra) ========

def lorentzian(x,A,mu,sig,N):
  numerator=A*0.5*sig/np.pi
  denominator=(x-mu)**2+(0.5*sig)**2
  return numerator/denominator +N

def gaussian(x,A,mu,sig,N):
  return N+A*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig,2.)))

def gaussian_based(x,A,mu,sig):
  return A*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig,2.)))

def power_law(x,i,a):
  return a*(x**-i)

def broken_power_law(x,i1,i2,a,f):
  p1=power_law(x,i1,a)
  p2=power_law(x,i2,a*(f**(i2-i1)))
  p1[x>f]=0
  p2[x<=f]=0
  return p1+p2
