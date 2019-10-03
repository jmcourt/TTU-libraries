import func_lib as func
import numpy as np
from numpy import random as rn
from scipy import fftpack as fou

def generate_signal_from_noise_func(function,maxf,minf=0,nf=100):
  freqs=np.arange(0,maxf,maxf/nf)
  amps=np.zeros(len(freqs),dtype=complex)
  is_even=len(freqs)%2==0
  for i in range(1,len(freqs)-1):
    f=freqs[i]
    re=rn.normal()
    im=rn.normal()*1j
    amp=(re+im)*function(f)
    amps[i]=amp
  f=freqs[-1]
  amp=rn.normal()
  if not is_even:
    amp+=rn.normal()*1j
  amps[len(freqs)-1]=amp*function(f)
  if is_even:
    conjpart=np.conj(amps[1:-1][::-1])
  else:
    conjpart=np.conj(amps[1:][::-1])
  amps=np.append(amps,conjpart)
  nyquist=freqs[len(freqs)-1]
  sample_rate=1/(2*nyquist)
  y=np.real(fou.ifft(amps))
  # ^ Real component should be 0, but machine errors mean its usually
  # around the 1 in 10^15 mark, so we can chuck it away
  x=np.arange(0,sample_rate*len(y),sample_rate)
  return x,y

def red_noise(index,maxf,minf=0,nf=100):
  def inp_func(fr):
    return func.power_law(fr,index,1)
  return generate_signal_from_noise_func(inp_func,maxf,minf=minf,nf=nf)

def broken_noise(index1,index2,breakf,maxf,minf=0,nf=100):
  def inp_func(fr):
    return func.broken_power_law(fr,index1,index2,1,breakf)
  return generate_signal_from_noise_func(inp_func,maxf,minf=minf,nf=nf)

def white_noise(maxf,minf=0,nf=100):
  return red_noise(1,maxf,minf=minf,nf=nf)


