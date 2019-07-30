import numpy as np
from numpy import random as rn
from scipy import fftpack as fou

from matplotlib import pyplot as pl

def red_noise(index,maxf,nf=100):
  # Red-Noise Lightcurve generator!  From Timmer & Konig 1995
  freqs=np.arange(0,maxf,maxf/nf)
  amps=np.zeros(len(freqs),dtype=complex)
  is_even=len(freqs)%2==0
  for i in range(1,len(freqs)-1):
    f=freqs[i]
    re=rn.normal()
    im=rn.normal()*1j
    amp=(re+im)*(f**-(index/2.))
    amps[i]=amp
  f=freqs[-1]
  amp=rn.normal()
  if not is_even:
    amp+=rn.normal()*1j
  amps[len(freqs)-1]=amp*(f**-(index/2.))
  if is_even:
    conjpart=np.conj(amps[1:-1][::-1])
  else:
    conjpart=np.conj(amps[1:][::-1])
  amps=np.append(amps,conjpart)
  nyquist=freqs[len(freqs)-1]
  pl.figure()
  pl.loglog(freqs,np.abs(amps**2)[:len(freqs)])
  pl.show(block=False)
  sample_rate=1/(2*nyquist)
  y=np.real(fou.ifft(amps))
  # ^ Real component should be 0, but machine errors mean its usually
  # around the 1 in 10^15 mark, so we can chuck it away
  x=np.arange(0,sample_rate*len(y),sample_rate)
  return x,y

def white_noise(maxf,nf=100):
  return red_noise(1,maxf,nf)
