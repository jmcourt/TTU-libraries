import numpy as np

def app_to_abs_map(mag,d):
  return mag-5*np.log10(d)+5

def abs_to_app_map(mag,d):
  return mag+5*np.log10(d)-5
