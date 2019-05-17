import numpy as np
from   scipy import interpolate as intp

class folder(object):
  def __init__(self,lc,period_info):    
    # Two different errors can be returned.  Default is the standard deviation of the distribution of y in each bin.  Otherwise can just propagate errors normally.
    self.x=lc.get_x()
    self.y=lc.get_y()
    self.ye=lc.get_ye()
    self.phases=self.calculate_p(self.x,period_info)#(x%period)/period
    self.period_info=period_info

  def fold(self,phase_bins,standev_errors=True):
    py=np.zeros(phase_bins)
    pye=np.zeros(phase_bins)
    for i in range(phase_bins):
      lower_phase=i/phase_bins
      upper_phase=(i+1)/phase_bins
      mask=np.logical_and(self.phases>=lower_phase,self.phases<upper_phase)
      py[i]=np.mean(self.y[mask])
      if standev_errors:
        pye[i]=np.std(self.y[mask])
      else:
        pye[i]=np.sqrt(np.sum(self.ye[mask]**2))/np.sum(mask)
    self.fx=np.arange(0,1,1/phase_bins)
    self.fy=py
    self.fye=pye
    self.phase_bins=phase_bins

  def calculate_p(self,x,period_info):
    raise NotImplementedError('No way to find period in base folder class')

  def get_p(self):
    return self.phases

  def get_x(self):
    return self.x
  def get_y(self):
    return self.y
  def get_ye(self):
    return self.ye

  def get_fx(self):
    return self.fx
  def get_fy(self):
    return self.fy
  def get_fye(self):
    return self.fye

class linear_folder(folder):
  def calculate_p(self,x,period):
    return (x%period)/period

class varifolder(folder):
  def calculate_p(self,x,zeros_list):
    p_phases=np.arange(0,len(zeros_list),1.0)
    self.spline=intp.PchipInterpolator(zeros_list, p_phases, extrapolate=True)
    phases=self.spline(x)
    phases=np.remainder(phases,1)
    return phases
