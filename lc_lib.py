import data_lib as dat
import file_lib as fi
import fold_lib as fo
import freq_lib as frq
import func_lib as func
from   matplotlib import gridspec as gs
from   matplotlib import pyplot as pl
import numpy as np
import random as rn
from   scipy import fftpack as fou
from   scipy import interpolate as intp
from   scipy import optimize as optm
from   scipy import signal as sgnl
import util_lib as util
import warnings as wr

# Optional packages

try:
  import astropy.coordinates as coord
  from   astropy.io import fits
  imported_astropy=True
except ImportError:
  wr.warn('No Astropy module found!  Unable to handle FITS files!')
  imported_astropy=False

try:
  from astroquery.simbad import Simbad
  from requests          import exceptions as rx
  imported_astroquery=True
except:
  wr.warn('No Astroquery module found!  Unable to query Simbad for object identifiers!')
  imported_astroquery=False

try:
  from loess import loess_1d as loe
  imported_loess=True
except:
  wr.warn('No Loess module found!  Unable to perform loess smoothing!')
  imported_loess=False

# ========= BASE LIGHTCURVE OBJECT! =====================================================================================================================

class lightcurve(dat.DataSet):

  # Basic initialisation of all lightcurves.  Gets x,y,ye and meta, and attempts to work out units

  @fi.mjit()
  def __init__(self,x,y,ye,meta={}):
    tx=np.array(x).astype(float)
    ty=np.array(y).astype(float)
    tye=np.array(ye).astype(float)
    nanmask=np.logical_not(np.logical_and(np.isnan(ty),np.isnan(tye)))
    tx=tx[nanmask]
    ty=ty[nanmask]
    tye=tye[nanmask]
    #sort data by time
    sorted_args=tx.argsort()
    self.set_x(tx[sorted_args])
    self.set_y(ty[sorted_args])
    self.set_ye(tye[sorted_args])

    if 'acceptable_gap' not in meta.keys():  # the gap width before a data gap is declared
      meta['acceptable_gap']=1.5
    self.set_acceptable_gap(meta['acceptable_gap']) # anything more than 1.5 times the median time separation is considered a data gap
    self.setup_meta(meta)
    if not self.has('t_units'):
      self.set_t_units('')
    if not self.has('y_units'):
      self.set_y_units('')
    if self.is_empty():
      self.set_binsize(0)
    else:
      self.set_binsize(min(self.delta_T()))
    self.ft_norm='N/A'
    self.unpack_metadata()
    self.set_folded_flag(False)
    self.set_data_even_flag(False)
    self.set_xphase_flag(False)
    self.set_period(0)
    self.period_calculated=False

  # Basic getters & setters

  def get_x(self):
    return self.x
  def get_y(self):
    return self.y
  def get_ye(self):
    return self.ye
  def set_x(self,x):
    self.x=x
  def set_y(self,y):
    self.y=y
  def set_ye(self,ye):
    self.ye=ye

  def get_bx(self):
    return self.bx
  def get_b(self):
    return self.b
  def get_be(self):
    return self.be
  def set_bx(self,bx):
    self.bx=bx
  def set_b(self,b):
    self.b=b
  def set_be(self,be):
    self.be=be

  def get_binsize(self):
    return self.binsize
  def set_binsize(self,binsize):
    self.binsize=binsize
  def get_period(self):
    return self.period
  def set_period(self,period):
    self.period=float(period)
    self.period_calculated=True
  def fetch_period(self):
    if self.has_period():
      return self.get_period()
    else:
      raise dat.DataError('No Period Specified!')
  def has_period(self):
    return self.period_calculated

  def get_t_units(self):
    return self.t_units
  def get_y_units(self):
    return self.y_units

  def set_t_units(self,unit):
    self.t_units=str(unit)
  def set_y_units(self,unit):
    self.y_units=str(unit)

  def get_length(self):
    return len(self.get_x())
  def get_len(self):
    return self.get_length()
  def get_acceptable_gap(self):
    return self.acceptable_gap
  def get_xrange(self):
    if self.is_empty():
      return 0
    return self.get_x()[-1]-self.get_x()[0]
  def get_yrange(self):
    if self.is_empty():
      return 0
    return self.get_y()[-1]-self.get_y()[0]
  def get_start_time(self):
    if self.is_empty():
      return 0
    return self.get_x()[0]
  def get_end_time(self):
    if self.is_empty():
      return 0
    return self.get_x()[-1]

  def get_max_time(self):
    if self.is_empty():
      return 0
    return self.get_x()[np.argmax(self.get_y())]
  def get_min_time(self):
    if self.is_empty():
      return 0
    return self.get_x()[np.argmin(self.get_y())]

  def get_title(self):
    return self.objname

  def set_acceptable_gap(self,gap):
    self.acceptable_gap=gap

  def set_phase_zeros(self,zeros):
    self.phase_zeros=zeros
  def get_phase_zeros(self):
    return self.phase_zeros

  def get_ft_freqs(self):
    return self.ft_freqs
  def get_ft(self):
    return self.ft
  def get_ft_e(self):
    return self.ft_e
  def get_ft_nupnu(self):
    return self.get_ft()*self.get_ft_freqs()
  def get_ft_nupnu_e(self):
    return self.get_ft_e()*self.get_ft_freqs()
  def get_ft_norm(self):
    return self.ft_norm

  def set_ft_freqs(self,ft_freqs):
    self.ft_freqs=ft_freqs
  def set_ft(self,ft):
    self.ft=ft
  def set_ft_e(self,ft_e):
    self.ft_e=ft_e
  def set_ft_norm(self,norm):
    self.ft_norm=str(norm)

  def get_ls_freqs(self):
    return self.ls_freqs
  def get_ls(self):
    return self.ls
  def get_ls_e(self):
    return self.ls_e
  def get_ls_nupnu(self):
    return self.get_ls()*self.get_ls_freqs()
  def get_ls_nupnu_e(self):
    return self.get_ls_e()*self.get_ls_freqs()

  def set_ls_freqs(self,ls_freqs):
    self.ls_freqs=ls_freqs
  def set_ls(self,ls):
    self.ls=ls
  def set_ls_e(self,ls_e):
    self.ls_e=ls_e

  # Flag checkers:

  def is_folded(self):
    return self.folded_flag
  def set_folded_flag(self,boolean):
    self.folded_flag=boolean

  def is_data_evened(self):
    return self.data_even_flag
  def set_data_even_flag(self,boolean):
    self.data_even_flag=boolean

  def x_axis_is_phase(self):
    return self.xphase_flag
  def set_xphase_flag(self,boolean):
    self.xphase_flag=boolean


  # Get me a spline!

  def get_spline(self,kind='slinear',fill_value='extrapolate'):
    spline=intp.interp1d(self.get_x(),self.get_y(),kind=kind,fill_value=fill_value)
    return spline

  def get_err_spline(self,kind='slinear',fill_value='extrapolate'):
    spline=intp.interp1d(self.get_x(),self.get_ye(),kind=kind,fill_value=fill_value)
    return spline

  def get_bg_spline(self,kind='slinear',fill_value='extrapolate'):
    spline=intp.interp1d(self.get_bx(),self.get_b(),kind=kind,fill_value=fill_value)
    return spline

  def get_bg_err_spline(self,kind='slinear',fill_value='extrapolate'):
    spline=intp.interp1d(self.get_bx(),self.get_be(),kind=kind,fill_value=fill_value)
    return spline

  # Function fitting

  def fit(self,function,init):
    self.fit_function=function
    op=optm.curve_fit(function,self.get_x(),self.get_y(),init)
    self.fit_params=op[0]
    self.fit_params_e=np.sqrt(np.diag(op[1]))

  # Dump contents to a csv

  def dump(self,filename,data_sep=' ',meta_sep=':'):
    k=list(self.get_meta_keys())
    k.sort()
    f=open(filename,'w')
    f.write('META_DATA\n')
    for key in k:
      if type(self.get_meta(key)) in (str,int,float):
        f.write(key+meta_sep+str(self.get_meta(key))+'\n')
    f.write('SCIENCE_DATA\n')
    for i in range(self.get_length()):
      f.write(str(self.get_x()[i])+data_sep+str(self.get_y()[i])+data_sep+str(self.get_ye()[i])+'\n')
    f.close()

  # Play with time axis

  def zero_time(self):
    if not self.is_empty():
      start_time=self.get_start_time()
      # Should be able to zero a lightcurve of length 0.  This matters for dynamic spectra
    else:
      start_time=0
    self.add_time(-start_time)

  def zeroed_time(self):
    s=self.copy()
    s.zero_time()
    return s

  def add_time(self,time):
    self.set_x(self.get_x()+time)
    self.shift_gtis(time)
    if self.has('b'):
      self.set_bx(self.get_bx()+time)

  def added_time(self,time):
    s=self.copy()
    s.add_time(time)
    return s

  def multiply_time(self,constant):
    self.set_x(self.get_x()*constant)
    self.set_binsize(self.get_binsize()*constant)
    self.multiply_gtis(constant)
    if self.has('b'):
      self.set_bx(self.get_bx()*constant)
    if self.has('ft_freqs'):
      self.set_ft_freqs(self.get_ft_freqs()/constant)
    if self.has('ls_freqs'):
      self.set_ls_freqs(self.get_ls_freqs()/constant)

  def multiplied_time(self,constant):
    s=self.copy()
    s.multiply_time(constant)
    return s

  def convert_days_to_s(self):
    self.multiply_time(86400)
    self.set_t_units('s')

  def convert_s_to_days(self):
    self.multiply_time(1./86400.)
    self.set_t_units('days')

  def converted_days_to_s(self):
    s=self.copy()
    s.convert_days_to_s()
    return s

  def converted_s_to_days(self):
    s=self.copy()
    s.convert_s_to_days()
    return s

  def shift_gtis(self,shift):
    pass    # Placeholder function to allow GTIs to be updated when data is renormed in objects such as RXTE lcs which store this information

  def multiply_gtis(self,constant):
    pass    # Placeholder function to allow GTIs to be updated when data is renormed in objects such as RXTE lcs which store this information

  # Self-explanatory quick-and-dirty plot machine.  BG plot checks if bg data is available, and dies if not

  def quickplot(self,output=None,errors=True,block=False,title=True,**kwargs):
    if self.is_folded():
      x=np.append(self.get_x(),self.get_x()+1)
      y=np.append(self.get_y(),self.get_y())
      ye=np.append(self.get_ye(),self.get_ye())
      xlab='Phase'
    else:
      x=self.get_x()
      y=self.get_y()
      ye=self.get_ye()
      xlab='Time '+self.t_unit_string()
    ax=fi.filter_axes(output)
    if title:
      ax.set_title(self.get_title()+' Quick Plot')
    if errors:
      ax.errorbar(x,y,yerr=ye,**kwargs)
    else:
      ax.plot(x,y,**kwargs)
    ax.set_xlabel(xlab)
    ax.set_ylabel('Rate '+self.y_unit_string())
    fi.plot_save(output,block)

  def plot_bg(self,output=None,block=False,**kwargs):
    if self.has('b'):
       raise NotImplementedError('No background data available in '+str(self.__class__)+' object')
    ax=fi.filter_axes(output)
    ax.set_title(self.get_title()+' bg Quick Plot')
    ax.errorbar(self.get_bx(),self.get_b(),yerr=self.get_be(),label='bg',**kwargs)
    ax.errorbar(self.get_x(),self.get_y(),yerr=self.get_ye(),label='phot',**kwargs)
    ax.legend()
    fi.plot_save(output,block)

  # slightly more adaptable plotting machine which takes dict inputs to give separate kwargs to errorbars and main line

  def plot_lc(self,output=None,block=False,title=True,xlabel=True,ylabel=True,kwargs={'color':'k'},ekwargs={'color':'0.7'}):
    if self.is_folded():
      x=np.append(self.get_x(),self.get_x()+1)
      y=np.append(self.get_y(),self.get_y())
      ye=np.append(self.get_ye(),self.get_ye())
      xlab='Phase'
    else:
      x=self.get_x()
      y=self.get_y()
      ye=self.get_ye()
      xlab='Time '+self.t_unit_string()
    ax=fi.filter_axes(output)

    if title==True:
      ax.set_title(self.get_title()+' LC Plot')
    elif title==False:
      pass
    else:
      ax.set_title(title)

    if xlabel==True:
      ax.set_xlabel(xlab)
    elif xlabel==False:
      pass
    else:
      ax.set_xlabel(xlabel)

    if ylabel==True:
      ax.set_ylabel('Rate '+self.y_unit_string())
    elif ylabel==False:
      pass
    else:
      ax.set_ylabel(ylabel)

    ax.errorbar(x,y,yerr=ye,zorder=-1,**ekwargs)
    ax.plot(x,y,zorder=0,**kwargs)
    fi.plot_save(output,block)

  # Creates a scatter plot of an unfolded lightcurve where the x-coord of each point is its phase

  def plot_folded_scatterplot(self,period=None,output=None,block=False,clip_percentile=100,**kwargs):
    if self.is_folded():
      wr.warn("Can't fold that which is already folded!")
    else:
      if period==None:
        period=self.fetch_period()
      folder=fo.linear_folder(self,period)
      ax=fi.filter_axes(output)
      p=folder.get_p()
      ax.scatter(np.append(p,p+1),np.append(self.get_y(),self.get_y()),marker='.',alpha=min(500/self.get_length(),0.2),color='k',**kwargs)
      ax.set_xlabel('Phase')
      ax.set_ylabel('Rate '+self.y_unit_string())
      ax.set_ylim(np.percentile(self.get_y(),100-clip_percentile),np.percentile(self.get_y(),clip_percentile))
      fi.plot_save(output,block)

  def plot_varifolded_scatterplot(self,zero_list,output=None,block=False,**kwargs):
    if self.is_folded():
      wr.warn("Can't fold that which is already folded!")
    else:
      folder=fo.varifolder(self,zero_list)
      ax=fi.filter_axes(output)
      p=folder.get_p()
      ax.scatter(np.append(p,p+1),np.append(self.get_y(),self.get_y()),marker='.',alpha=min(500/self.get_length(),0.2),color='k',**kwargs)
      ax.set_xlabel('Phase')
      ax.set_ylabel('Rate '+self.y_unit_string())
      fi.plot_save(output,block)

  # return approx location of significant data gaps (>25* median time separation by default)

  def delta_T(self):
    return self.get_x()[1:]-self.get_x()[:-1]

  def get_data_gaps(self):
    delta_T=self.delta_T()
    acceptable_gap=self.get_acceptable_gap()*np.median(delta_T)
    gap_mask=delta_T>acceptable_gap
    gap_starts=self.get_x()[:-1][gap_mask]
    gap_ends=self.get_x()[1:][gap_mask]
    gaps=[(gap_starts[i],gap_ends[i]) for i in range(len(gap_starts))]
    return gaps

  # RMS and associated plots

  def get_rms(self):
    return rms(self.get_y())

  def prep_variability_stats(self,window):
    n_windows=int(self.get_xrange()/window)
    start=self.get_start_time()
    rx=np.zeros(n_windows)
    ry=np.zeros(n_windows)
    rr=np.zeros(n_windows)-1
    rp=np.zeros(n_windows)
    rd=np.zeros(n_windows)
    data_gaps=self.get_data_gaps()
    for i in range(n_windows):
      w_start=start+i*window
      w_end  =start+(i+1)*window
      if is_overlap((w_start,w_end),data_gaps):
        continue
      section=self.calved(w_start,w_end)
      rx[i]=section.get_start_time()
      ry[i]=section.get_mean()
      rr[i]=section.get_rms()  # NOTE, the RMS values are in RR, NOT RY!!  RY is for meaned stuff!
      rp[i]=section.get_max()-section.get_mean()
      rd[i]=section.get_min()-section.get_mean()
    mask=rr>=0  # RMS must be positive!  Also filters out all the skipped points cause of that cheeky -1 on the rr assignment line
    self.rms_over_time_x=rx[mask]
    self.rms_over_time_y=np.abs(ry)[mask]
    self.rms_over_time_r=rr[mask]
    self.rms_over_time_peak_heights=rp[mask]
    self.rms_over_time_dip_depths=rd[mask]

  def plot_rms(self,fractional=False,output=None,x_unit='time',block=False,**kwargs):
    if not self.has('rms_over_time_x'):
      raise dat.DataError('RMS plots not prepared!  Prepare with prep_variability_stats')
    if x_unit.lower() not in ('time','rate'):
      raise dat.DataError('I dont recognise the X-unit "'+str(x_unit)+'"!  Valid X-units are "time" or "rate"')
    ax=fi.filter_axes(output)
    ax.set_title(self.get_title()+' RMS over '+x_unit.title()+' Plot')
    if fractional:
      y=self.rms_over_time_r/self.rms_over_time_y
      ax.set_ylabel('Fractional RMS')
    else:
      y=self.rms_over_time_r
      ax.set_ylabel('RMS')
    if x_unit.lower()=='time':
      ax.set_xlabel('Time '+self.t_unit_string())
      ax.plot(self.rms_over_time_x,y,**kwargs)
    else:
      ax.set_xlabel('Rate '+self.y_unit_string())
      ax.scatter(self.rms_over_time_y,y,**kwargs)
    fi.plot_save(output,block)

  def plot_peak_heights(self,fractional=False,output=None,x_unit='time',block=False,**kwargs):
    if not self.has('rms_over_time_x'):
      raise dat.DataError('Peak heights plot not prepared!  Prepare with prep_variability_stats')
    if x_unit.lower() not in ('time','rate'):
      raise dat.DataError('I dont recognise the X-unit "'+str(x_unit)+'"!  Valid X-units are "time" or "rate"')
    ax=fi.filter_axes(output)
    ax.set_title(self.get_title()+' Peak Heights over '+x_unit.title()+' Plot')
    if fractional:
      y=self.rms_over_time_peak_heights/self.rms_over_time_y
      ax.set_ylabel('Fractional Max Peak Height')
    else:
      y=self.rms_over_time_peak_heights
      ax.set_ylabel('Max Peak Height')
    if x_unit.lower()=='time':
      ax.set_xlabel('Time '+self.t_unit_string())
      ax.plot(self.rms_over_time_x,y,**kwargs)
    else:
      ax.set_xlabel('Rate '+self.y_unit_string())
      ax.scatter(self.rms_over_time_y,y,**kwargs)
    fi.plot_save(output,block)

  def plot_dip_depths(self,fractional=False,output=None,x_unit='time',block=False,**kwargs):
    if not self.has('rms_over_time_x'):
      raise dat.DataError('Dip depths plot not prepared!  Prepare with prep_variability_stats')
    if x_unit.lower() not in ('time','rate'):
      raise dat.DataError('I dont recognise the X-unit "'+str(x_unit)+'"!  Valid X-units are "time" or "rate"')
    ax=fi.filter_axes(output)
    ax.set_title(self.get_title()+' Dip Depths over '+x_unit.title()+' Plot')
    if fractional:
      y=self.rms_over_time_dip_depths/self.rms_over_time_y
      ax.set_ylabel('Fractional Max Dip Depth')
    else:
      y=self.rms_over_time_dip_depths
      ax.set_ylabel('Max Dip Depth')
    if x_unit.lower()=='time':
      ax.set_xlabel('Time '+self.t_unit_string())
      ax.plot(self.rms_over_time_x,y,**kwargs)
    else:
      ax.set_xlabel('Rate '+self.y_unit_string())
      ax.scatter(self.rms_over_time_y,y,**kwargs)
    fi.plot_save(output,block)

  def plot_peaks_and_dips(self,fractional=False,output=None,x_unit='time',block=False,**kwargs):
    if not self.has('rms_over_time_x'):
      raise dat.DataError('Dip depths plot not prepared!  Prepare with prep_variability_stats')
    if x_unit.lower() not in ('time','rate'):
      raise dat.DataError('I dont recognise the X-unit "'+str(x_unit)+'"!  Valid X-units are "time" or "rate"')
    ax=fi.filter_axes(output)
    ax.set_title(self.get_title()+' Peak Heights & Dip Depths over '+x_unit.title()+' Plot')
    if fractional:
      y1=self.rms_over_time_peak_heights/self.rms_over_time_y
      y2=self.rms_over_time_dip_depths/self.rms_over_time_y
      ax.set_ylabel('Fractional Height/Depth')
    else:
      y1=self.rms_over_time_peak_heights
      y2=self.rms_over_time_dip_depths
      ax.set_ylabel('Max Height/Depth')
    if x_unit.lower()=='time':
      ax.set_xlabel('Time '+self.t_unit_string())
      ax.plot(self.rms_over_time_x,y1,label='Peak Heights',**kwargs)
      ax.plot(self.rms_over_time_x,y2,label='Dip Depths',**kwargs)
      ax.fill_between(self.rms_over_time_x,y1,y2,color='0.7')
      ax.axhline(0,color='k')
      ax.legend()
    else:
      ax.set_xlabel('Rate '+self.y_unit_string())
      ax.scatter(self.rms_over_time_y,y1,label='Peak Heights',**kwargs)
      ax.scatter(self.rms_over_time_y,y2,label='Dip Depths',**kwargs)
      ax.axhline(0,color='k')
      ax.legend()
    fi.plot_save(output,block)
    
  # Some general Fourier methods

  def get_nyquist(self):
    return 0.5/self.get_binsize()

  def fourier(self,norm='leahy',normname='custom',squelch=False,const=2):
    if not squelch and not self.is_data_evened():
      wr.warn('Internal Fourier method in lightcurve objects does NOT check for evenly spaced data yet!')
    try:
      norm=float(norm)
      custnorm=True
    except ValueError:
      norm=norm.lower()
      custnorm=False
    raw_ft=fou.fft(self.get_y())
    ft=np.abs(raw_ft[1:self.get_length()//2])**2  # crop all data above nyquist and obtain the amplitudes of the FT
    if custnorm:
      self.set_ft_norm(normname)
      self.set_ft(ft*norm)
    elif norm=='leahy':
      self.set_ft_norm('leahy')
      self.set_ft(frq.leahy(ft,np.sum(self.get_y())))
    elif norm=='rms':
      if not self.has('b'): raise dat.DataError('No background data available to RMS normalise FT in '+str(self.__class__)+' object')
      self.set_ft_norm('rms')
      self.set_ft(frq.rms(ft,np.sum(self.get_y()),np.mean(self.get_y()+self.get_b()),np.mean(self.get_b*()),const=const))
    else:
      if norm!='none' and not squelch:
        wr.warn('Invalid Fourier normalisation '+norm+' specified: using None normalisation')
      self.set_ft_norm('none')
      self.set_ft(ft)
    freqs=np.linspace(0,self.get_nyquist(),len(ft)+1)[1:]
    self.set_ft_e(self.get_ft())
    self.set_ft_freqs(freqs)

  def windowed_fourier(self,window_size,norm='leahy',normname='custom',const=2,squelch=False):
    if not squelch and not self.is_data_evened():
      wr.warn('Internal Fourier method in lightcurve objects does NOT check for evenly spaced data yet!')
    npts=int(window_size/self.get_binsize())
    lognpoints=np.log2(npts)
    npts=2**round(lognpoints)
    init_arrays=False
    new_window_size=npts*self.get_binsize()
    if np.abs(new_window_size-window_size)>0.1*window_size and not squelch:
      print('Window size rounded to '+str(new_window_size)+' '+self.get_t_units())
    zero=self.get_start_time()
    data_gaps=self.get_data_gaps()
    for win_i in range(0,int(self.get_xrange()/new_window_size)):
      st=zero+win_i*new_window_size
      ed=zero+(win_i+1)*new_window_size
      if is_overlap((st,ed),data_gaps):
        continue
      window=self.calved(st,ed)
      window.fourier(norm=norm,normname=normname,const=const,squelch=True)
      if not init_arrays:
        init_arrays=True
        total=1
        self.set_ft_freqs(window.get_ft_freqs())
        self.set_ft_norm(window.get_ft_norm())
        ft=window.get_ft()
        ft_e=window.get_ft_e()**2
      else:
        total+=1
        ft+=window.get_ft()
        ft_e+=window.get_ft_e()**2
    self.set_ft(ft/total)
    self.set_ft_e(np.sqrt(ft_e)/total)

  def rebin_fourier(self,bin_factor):
    if not self.has('ft'):
      raise dat.DataError('No Fourier spectrum to rebin!')
    newft_freqs,newft,newft_e=dat.rebin(bin_factor,self.get_ft_freqs(),self.get_ft(),ye=self.get_ft_e())
    self.set_ft(newft)
    self.set_ft_freqs(newft_freqs)
    self.set_ft_e(newft_e)

  def log_rebin_fourier(self,log_res):
    if not self.has('ft'):
      raise dat.DataError('No Fourier spectrum to rebin!')
    newft_freqs,newft,newft_e=dat.log_rebin(log_res,self.get_ft_freqs(),self.get_ft(),ye=self.get_ft_e())
    self.set_ft(newft)
    self.set_ft_freqs(newft_freqs)
    self.set_ft_e(newft_e)
    
  def plot_fourier(self,output=None,log=False,logx=False,nupnu=False,**kwargs):
    if not self.has('ft'): self.fourier('leahy')
    ax=fi.filter_axes(output)
    ax.set_title(self.get_title()+' Fourier Spectrum')
    if log:
      ax.semilogy()
    if logx:
      ax.semilogx()
    if nupnu:
      ax.errorbar(self.get_ft_freqs(),self.get_ft_nupnu(),yerr=self.get_ft_nupnu_e(),**kwargs)
    else:
      ax.errorbar(self.get_ft_freqs(),self.get_ft(),yerr=self.get_ft_e(),**kwargs)
    ax.set_ylabel('"'+self.get_ft_norm()+'"-normalised power')
    ax.set_xlabel('Frequency ('+self.get_t_units()+'^-1)')
    fi.plot_save(output,**kwargs)

  def fourier_spectrogram(self,binsize,bin_separation):
    raise NotImplementedError('Fourier Spectrogram method not coded yet!')

  # Some general L-S methods

  def lomb_scargle(self,freqrange,norm='auto',generalised=True,squelch=False,errors=False,bootstrapN=100):

    # Generalised L-S from Zechmeister & Kuerster, 2009, eq 5-15

    if len(freqrange)>9999 and not squelch:
      wr.warn(str(len(freqrange))+' frequency samples requested!  This could take a while.')
      if not util.proceed():
        return None
    self.set_ls(frq.lomb_scargle(self.get_x(),self.get_y(),self.get_ye(),freqrange,norm=norm,generalised=generalised))
    self.set_ls_freqs(np.array(freqrange))
    if errors:
      errgrid=np.zeros((len(freqrange),bootstrapN))
      for i in range(bootstrapN):
        sample=self.copy()
        fake_y=dat.gaussian_bootstrap(sample.get_y(),sample.get_ye())
        sample.set_y(fake_y)
        sample.lomb_scargle(freqrange,norm=norm,generalised=generalised,squelch=True,errors=False)
        errgrid[:,i]=sample.get_ls()
      self.set_ls_e(np.std(errgrid,axis=1))
    else:
      self.set_ls_e(self.get_ls()*0)

  def auto_lomb_scargle(self,min_f=None,max_f=None,resolution=None,n0=1,norm='auto',generalised=True,squelch=False,errors=False):
    # n0 is the oversampling ratio.  5 is reccommended by e.g. Schwarzenberg-Czerny 1996
    if min_f==None:
      min_f=1/self.get_xrange()
    if max_f==None:
      max_f=self.get_nyquist()
    if resolution==None:
      resolution=1/(n0*self.get_xrange())
    f_range=np.arange(min_f,max_f,resolution)
    self.lomb_scargle(f_range,norm=norm,generalised=generalised,squelch=squelch,errors=errors)

  def show_lomb_scargle_limits(self):
    print('Min Frequency: '+str(1/self.get_xrange()))
    print('Max Frequency: '+str(self.get_nyquist()))
    print('Resolution   : '+str(1/(self.get_xrange())))

  def plot_lomb_scargle(self,log=False,logx=False,nupnu=False,output=None,as_period=False,block=False,errors=False,**kwargs):
    if not self.has('ls'): self.auto_lomb_scargle()
    ax=fi.filter_axes(output)
    ax.set_title(self.get_title()+' Lomb-Scargle Periodogram')
    if as_period:
      x=1/self.get_ls_freqs()
    else:
      x=self.get_ls_freqs()
    if log:
      ax.semilogy()
    if logx:
      ax.semilogx()
    if errors:
      if nupnu:
        ax.errorbar(x,self.get_ls_nupnu(),yerr=self.get_ls_nupnu_e(),**kwargs)
      else:
        ax.errorbar(x,self.get_ls(),yerr=self.get_ls_e(),**kwargs)
    else:
      if nupnu:
        ax.plot(x,self.get_ls_nupnu(),**kwargs)
      else:
        ax.plot(x,self.get_ls(),**kwargs)
    ax.set_ylabel('Lomb-Scargle power')
    if as_period:
      ax.set_xlabel('Period ('+self.get_t_units()+')')
    else:
      ax.set_xlabel('Frequency ('+self.get_t_units()+'^-1)')
    fi.plot_save(output,block)

  def rebin_lomb_scargle(self,bin_factor):
    if not self.has('ls'):
      raise dat.DataError('No Lomb-Scargle spectrum to rebin!')
    newls_freqs,newls,newls_e=dat.rebin(bin_factor,self.get_ls_freqs(),self.get_ls(),ye=self.get_ls_e())
    self.set_ls(newls)
    self.set_ls_e(newls_e)
    self.set_ls_freqs(newls_freqs)

  def log_rebin_lomb_scargle(self,log_res):
    if not self.has('ls'):
      raise dat.DataError('No Lomb-Scargle spectrum to rebin!')
    newls_freqs,newls,newls_e=dat.log_rebin(log_res,self.get_ls_freqs(),self.get_ls(),ye=self.get_ls_e())
    self.set_ls(newls)
    self.set_ls_e(newls_e)
    self.set_ls_freqs(newls_freqs)

  def get_freq_resolution(self): # get the minimum frequency resolution for LS or Fourier without oversampling
    minf=1/self.get_xrange()
    maxf=self.get_nyquist()
    ndat=self.get_length()
    return (maxf-minf)/ndat

  def fit_qpo(self,f_min,f_max,plot=False,**kwargs):
    frange=f_max-f_min
    newlc=self.copy()
    newlc.lomb_scargle(np.arange(f_min,f_max,self.get_freq_resolution()))
    newlc.plot_lomb_scargle()
    init_vals=(np.max(newlc.get_ls_freqs()),f_min+frange/2,frange/2,0)
    fit_results=optm.curve_fit(func.lorentzian,newlc.get_ls_freqs(),newlc.get_ls(),init_vals)
    if plot:
      pl.figure()
      ax=pl.gca()
      newlc.plot_lomb_scargle(output=ax)
      plotrange=np.arange(newlc.get_ls_freqs()[0],newlc.get_ls_freqs()[-1],(newlc.get_ls_freqs()[-1]-newlc.get_ls_freqs()[0])/100)
      ax.plot(plotrange,func.lorentzian(plotrange,*fit_results[0]),':k',**kwargs)
      pl.show()
    return fit_results

  def lomb_scargle_spectrogram(self,freqrange,binsize,bin_separation,ignore_gaps=True,as_return=False,squelch=False):
    min_points=80
    progress=0
    starttime=self.get_start_time()
    numbins=max(int(((self.get_xrange())-binsize)//bin_separation),0)
    if numbins==0:
      raise ValueError('Bin width longer than data set!')
    lsnorm=(self.get_length()-1)/(2.0*numbins)
    dynamic_spectrum=np.zeros((numbins,len(freqrange)))
    N_datapoints=[np.nan]*numbins
    data_gaps=self.get_data_gaps()
    for i in range(numbins):
      if not squelch:
        new_progress=int(i/numbins*100)
        if new_progress>progress:
          print('Lomb_Scargle Periodogram: '+str(progress)+'% complete')
          progress=new_progress
      calve_st=starttime+i*bin_separation
      calve_ed=calve_st+binsize
      if ignore_gaps:
        if is_overlap((calve_st,calve_ed),data_gaps):
          continue
      calved_lc=self.calved(calve_st,calve_ed)
      calved_lc.lomb_scargle(freqrange,norm=lsnorm)
      lomb_scargle_spec=calved_lc.get_ls()
      dynamic_spectrum[i,:]=lomb_scargle_spec
      N_datapoints[i]=calved_lc.get_length()
    dynamic_ls_spectrum         = dynamic_spectrum.T
    dynamic_ls_spectrum_tvalues = starttime+bin_separation*(np.arange(0,numbins,1.0))+binsize/2.0
    dynamic_ls_spectrum_fvalues = freqrange
    dmeta={'x_dimension':'Time','y_dimension':'Frequency','x_unit':self.get_t_units(),'y_unit':self.get_y_units()}
    dls_data=dat.TwoD_Dataframe(dynamic_ls_spectrum_tvalues,dynamic_ls_spectrum_fvalues,dynamic_ls_spectrum,meta=dmeta)
    if as_return:
      return dls_data,N_datapoints
    self.dynamic_ls_data=dls_data

  def windowed_lomb_scargle(self,freqrange,window_size,ignore_gaps=True):
    ls_array,ls_datapoints=self.lomb_scargle_spectrogram(freqrange,window_size,window_size,ignore_gaps=ignore_gaps,as_return=True,squelch=True)
    data=ls_array.get_z()
    data[np.isnan(data)]=0
    ls_datapoints=np.array(ls_datapoints)
    ls_datapoints[np.isnan(ls_datapoints)]=0 # turn NaNs into zeros to remove them from weighting
    weighted_data=np.multiply(ls_datapoints,data)
    weighted_errs=np.multiply(ls_datapoints,data**2)
    sumN=np.sum(ls_datapoints)
    self.set_ls(np.sum(weighted_data,axis=1)/sumN)
    self.set_ls_freqs(np.array(freqrange))
    self.set_ls_e(np.sqrt(np.sum(weighted_errs,axis=1))/sumN)

  def dump_lomb_scargle_spectrogram(self,filename,header=True):
    if not self.has('dynamic_ls_data'):
      wr.warn('Dynamic Lomb-Scargle Spectrum not prepared!  Skipping csv dump!')
      return None
    self.dynamic_ls_data.dump(filename,header=header)

  def get_lomb_scargle_spectrogram(self):
    if not self.has('dynamic_ls_data'):
      wr.warn('Dynamic Lomb-Scargle Spectrum not prepared!  Skipping grabbing!')
    return self.dynamic_ls_data

  def plot_lomb_scargle_spectrogram(self,colour_range='auto',output=None,block=False,cmap='viridis',colourbar=True,cbar_ax=None,**kwargs):
    if not self.has('dynamic_ls_data'):
      raise dat.DataError('Dynamic Lomb-Scargle Spectrum not prepared!')
    elif self.dynamic_ls_data.get_max()<=0:
      raise dat.DataError('Dynamic LS spectrum is empty!  Is your Acceptable Gap parameter too small?')
    ax=fi.filter_axes(output)
    ax.set_title(self.get_title()+' Dynamic Lomb-Scargle Periodogram')
    ax.set_xlabel('Time '+self.t_unit_string())
    ax.set_ylabel('Frequency ('+self.get_t_units()+'^-1)')
    Z=self.dynamic_ls_data.get_z()
    if colour_range=='auto':
      colour_min=np.min(Z[Z>0])
      colour_max=np.max(Z)
    else:
      colour_min=colour_range[0]
      colour_max=colour_range[1]
    c=self.dynamic_ls_data.log_colour_plot(colour_min=colour_min,colour_max=colour_max,ax=ax,cmap=cmap)
    if colourbar:
      if cbar_ax==None:
        c=pl.colorbar(c,ax=ax)
      else:
        c=pl.colorbar(c,cax=cbar_ax)
    fi.plot_save(output,block)
    return c

  def plot_lomb_scargle_spectrogram_combo(self,colour_range='auto',filename=None,with_lc=True,with_1d_ls=True,figsize=(20,20),cmap='viridis',block=False):

    if not self.has('dynamic_ls_data'):
      wr.warn('Dynamic Lomb-Scargle Spectrum not prepared!  Skipping plotting!')
      return None
    elif self.dynamic_ls_data.get_max()<=0:
      wr.warn('Dynamic LS spectrum is empty!  Is your Acceptable Gap parameter too small?')
      return None
    if with_1d_ls:
      self.lomb_scargle(self.dynamic_ls_data.get_y())  # if the user wants a 1D LS plotted but hasnt made one yet, make one with the same params as the 2D LS

    size_ratio=4
    grid=gs.GridSpec(size_ratio+with_lc,size_ratio+with_1d_ls)

    fig=pl.figure(figsize=figsize)
    ax_main=fig.add_subplot(grid[:size_ratio,:size_ratio])
    self.plot_lomb_scargle_spectrogram(output=ax_main,colour_range=colour_range,colourbar=(not with_lc) and (not with_1d_ls))
    if with_lc:
      ax_main.set_xlabel('')
    if with_lc:
      ax_main.set_xticks([])
      ax_lc=fig.add_subplot(grid[size_ratio,:size_ratio])
      self.quickplot(output=ax_lc)
      ax_lc.set_title('')
      ax_lc.set_ylabel('Rate '+self.y_unit_string())
      ax_lc.set_xlim(min(self.dynamic_ls_data.get_x()),max(self.dynamic_ls_data.get_x()))
      ax_lc.set_xlabel('Time '+self.t_unit_string())
    if with_1d_ls:
      ax_ls=fig.add_subplot(grid[:size_ratio,size_ratio])
      ax_ls.semilogx(self.get_ls(),self.get_ls_freqs(),'k')
      ax_ls.set_yticks([])
      ax_ls.set_xlabel('L-S Power')
      ax_ls.fill_betweenx(self.get_ls_freqs(),self.get_ls(),0,facecolor='0.7')
      ax_ls.set_xlim(np.percentile(self.get_ls(),25),max(self.get_ls()*1.01))
      ax_ls.set_ylim(min(self.dynamic_ls_data.get_y()),max(self.dynamic_ls_data.get_y()))
    fi.plot_save(filename,block)

  # Add data from a matching lc object

  def add_data(self,lc):
    if lc.__class__!=self.__class__:
      raise TypeError('Cannot concatenate '+str(lc.__class__)+' and '+str(self.__class__)+'!')
    dictkeys1=self.get_contents()
    dictkeys2=lc.get_contents()
    keyset=(set(dictkeys1)|set(dictkeys2))-set(('x','y','ye','bx','b','be','meta','sector')) #some protected dict items which are allowed to mismatch
    for key in keyset:
      if not self.has(key):
        self.set_contents(key,lc.get_contents([key]))
      elif not lc.has(key):
        pass
      elif self.get_contents(key)!=lc.get_contents(key):
        wr.warn('Warning: '+key.upper()+' does not match!')
    if self.has('b') and lc.has('b'):
      both_bg=True
    else:
      both_bg=False
    if lc.get_start_time()>self.get_end_time():
      self.set_x(np.append(self.get_x(),lc.get_x()))
      self.set_y(np.append(self.get_y(),lc.get_y()))
      self.set_ye(np.append(self.get_ye(),lc.get_ye()))
      if both_bg:
        self.set_bx(np.append(self.get_bx(),lc.get_bx()))
        self.set_b(np.append(self.get_b(),lc.get_b()))
        self.set_be(np.append(self.get_be(),lc.get_be()))
    elif lc.get_end_time()<self.get_start_time():
      self.set_x(np.append(lc.x,self.get_x()))
      self.set_y(np.append(lc.get_y(),self.get_y()))
      self.set_ye(np.append(lc.get_ye(),self.get_ye()))
      if both_bg:
        self.set_bx(np.append(lc.get_bx(),self.get_bx()))
        self.set_b(np.append(lc.get_b(),self.get_b()))
        self.set_be(np.append(lc.get_be(),self.get_be()))
    else:
      raise dat.DataError('Datasets overlap in time!  Cannot concatenate!')
    if self.has('sector'):
      self.sector=''  # nuke sector info if several sectors have been cated
      self.set_meta('sector','')

  def added_data(self,lc):
    s=self.copy()
    s.add_data(lc)
    return s

  # Plot label generation

  def t_unit_string(self):
    if self.get_t_units()=='':
      return ''
    else:
      return '('+self.get_t_units()+')'

  def y_unit_string(self):
    if self.get_y_units()=='':
      return ''
    else:
      return '('+self.get_y_units()+')'

  #### These functions have an in-place version, and a -ed version which returns a new object ####

  # Returns a subset of the lightcurve between t=stime and t=etime.  New lc retains class of parent

  def calve(self,stime,etime):
    with wr.catch_warnings():
      wr.filterwarnings('ignore',message='invalid value encountered in less')
      wr.filterwarnings('ignore',message='invalid value encountered in greater_equal')
      mask=np.logical_and(self.get_x()>=stime,self.get_x()<etime)
    self.set_x(self.get_x()[mask])
    self.set_y(self.get_y()[mask])
    self.set_ye(self.get_ye()[mask])
    if self.has('b'):
      with wr.catch_warnings():
        wr.filterwarnings('ignore',message='invalid value encountered in less')
        wr.filterwarnings('ignore',message='invalid value encountered in greater_equal')
        mask=np.logical_and(self.get_bx()>=stime,self.get_bx()<etime)
      self.set_bx(self.get_bx()[mask])
      self.set_b(self.get_b()[mask])
      self.set_be(self.get_be()[mask])

  def calved(self,stime,etime):
    s=self.copy()
    s.calve(stime,etime)
    return s

  def calve_by_length(self,length):
    self.set_x(self.get_x()[:length])
    self.set_y(self.get_y()[:length])
    self.set_ye(self.get_ye()[:length])

  def calved_by_length(self,length):
    s=self.copy()
    s.calve_by_length(length)
    return s

  # Attempts to detrend the data for a given window size.

  def detrend(self,window_size,method='None'):
    newy=self.get_y()-smart_smooth(self,window_size,method)
    self.set_y(newy)

  def detrended(self,window_size,method='None'):
    s=self.copy()
    s.detrend(window_size,method)
    return s

  def plot_with_trend(self,window_size,method='None',output=None,block=False,**kwargs):
    ax=fi.filter_axes(output)
    smoothed_lc=self.smoothed(window_size,method)
    self.quickplot(output=ax,**kwargs)
    smoothed_lc.quickplot(output=ax,errors=False,**kwargs)
    fi.plot_save(output,block)

  # smooths the data for a given window size.  Sorta the opposite of above.

  def smooth(self,parameter,method='None'):
    newy=smart_smooth(self,parameter,method)
    self.set_y(newy)

  def smoothed(self,parameter,method='None'):
    s=self.copy()
    s.smooth(window_size,parameter)
    return s

  # returns 2 lightcurves: the Smoothed and the Detrended

  def decomposed(self,parameter,method='None'):
    smoothed_y=smart_smooth(self,parameter,method)
    s1=self.copy()
    s2=self.copy()
    s1.set_y(smoothed_y)
    s2.set_y(s2.get_y()-smoothed_y)
    return s1,s2

  # Shuffler; returns the y,ye pairs in a random order

  def shuffle(self):
    indices=np.arange(self.get_length(),dtype=int)
    rn.shuffle(indices)
    self.set_y(self.get_y()[indices])
    self.set_ye(self.get_ye()[indices])

  def shuffled(self):
    s=self.copy()
    s.shuffle()
    return s

  # Bin evener & spline evener: forces data into bins of even width!
  def even_bins(self,binsize=None):
    if binsize==None: binsize=self.get_binsize()
    newx=np.arange(self.get_start_time(),self.get_end_time(),binsize)
    newy=np.zeros(len(newx))*np.nan
    newe=np.zeros(len(newx))*np.nan
    for i in range(len(newx)):
      b_left=newx[i]
      mask=np.logical_and(self.get_x()>=b_left,self.get_x()<b_left+binsize)
      N=np.sum(mask)
      if N==0:
        continue
      newy[i]=np.mean(self.get_y()[mask])
      newe[i]=np.sqrt(np.sum(self.get_ye()[mask]**2))/N
    self.set_x(newx)
    self.set_y(newy)
    self.set_ye(newe)
    self.set_binsize(binsize)

  def evened_bins(self,binsize=None):
    s=self.copy()
    s.even_bins(binsize)
    return s

  def spline_even(self,binsize=None):
    if binsize==None: binsize=self.get_binsize()
    gaps=self.get_data_gaps()
    newx=np.arange(self.get_start_time(),self.get_end_time(),binsize)

    evener_spline=self.get_spline()
    errorer_spline=self.get_err_spline()

    newy=evener_spline(newx)
    newye=errorer_spline(newx)

    if self.has('b'):
      evener_b_spline=self.get_bg_spline()
      errorer_b_spline=self.get_bg_err_spline()
      newby=evener_b_spline(newx)
      newbye=errorer_b_spline(newx)

    mask=np.ones(len(newx),dtype=bool)
    for gap in gaps:
      gapmask=np.logical_or(newx<gap[0],newx>gap[1])
      mask=np.logical_and(mask,gapmask)

    self.set_x(newx[mask])
    self.set_y(newy[mask])
    self.set_ye(newye[mask])
    self.set_binsize(binsize)

    if self.has('b'):
      self.set_bx(newx[mask])
      self.set_b(newby[mask])
      self.set_be(newbye[mask])

    self.set_data_even_flag(True)

  def spline_evened(self,binsize=None):
    s=self.copy()
    s.spline_even(binsize)
    return s

  # Rebinning algorithms.  How was this not already a thing!?

  def rebin(self,time):
    newx,newy,newye=dat.rebin(time,self.get_x(),self.get_y(),ye=self.get_ye())
    self.set_x(newx)
    self.set_y(newy)
    self.set_ye(newye)

  def rebinned(self,time):
    s=self.copy()
    s.rebin(time)
    return s

  def rebin_by_factor(self,factor):
    newx,newy,null,newye=dat.rebin_by_factor(factor,self.get_x(),self.get_y(),ye=self.get_ye())
    self.set_x(newx)
    self.set_y(newy)
    self.set_ye(newye)

  def rebinned_by_factor(self,factor):
    s=self.copy()
    s.rebin_by_factor(factor)
    return s

  # Eww yuck get rid of magnitude measurements

  def demagnitude(self):
    o_y=self.get_y()
    o_ye=self.get_ye()
    self.set_y((100**0.2)**-o_y)
    nerr=self.get_y()-((100**0.2)**-(o_y+o_ye))
    perr=((100**0.2)**-(o_y-o_ye))-self.get_y()
    self.set_ye((nerr*perr)**0.5)
    self.set_y_units('Vega')

  def demagnituded(self):
    s=self.copy()
    s.demagnitude()
    return s

  # Flux phase diagrams!

  def flux_phase_diagram(self,period=None,Ncycles_per_line=2,offset=None):
    if period==None:
      period=self.fetch_period()
    phase_bins=int(round(period/self.get_binsize())*Ncycles_per_line)
    if offset==0:
      offset=None
    if offset==None:
      npdata=self
    elif offset<0:
      wr.warn('Negative offset provided!  Setting offset to zero.')
      npdata=self
    else:
      npdata=self.nan_padded((1-offset)*period,front=True)
    even_data=npdata.evened_bins(period*Ncycles_per_line/phase_bins)
    length=even_data.get_length()
    length=(length//phase_bins)*phase_bins
    even_data.calve_by_length(length)
    n_columns=int(even_data.get_length()/phase_bins)
    ph_axis=np.arange(0,1,1/phase_bins)*Ncycles_per_line
    cy_axis=np.arange(0,n_columns)*Ncycles_per_line
    Z=even_data.get_y().reshape(n_columns,phase_bins)
    self.fp_data=dat.TwoD_Dataframe(ph_axis,cy_axis,Z)
    
  def plot_flux_phase_diagram(self,period=None,Ncycles_per_line=2,offset=None,output=None,block=False,norm_per_line=True,nans_as=np.nan,colour_range='auto',cmap='viridis',colourbar=False,cbar_ax=None):
    if not self.has('fp_data'):
      self.flux_phase_diagram(period,Ncycles_per_line,offset=offset)
    plot_data=self.fp_data.copy()
    ax=fi.filter_axes(output)
    ax.set_title(self.get_title()+' Flux-Phase Diagram')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Cycle #')
    if norm_per_line:
      plot_data.xdir_norm()
    if nans_as=='mean':
      nans_as=plot_data.get_mean()
    elif nans_as=='min':
      nans_as=plot_data.get_min()
    plot_data.z[np.isnan(plot_data.z)]=nans_as
    if colour_range=='auto':
      colour_min=plot_data.get_min()
      colour_max=plot_data.get_max()
    else:
      colour_min=colour_range[0]
      colour_max=colour_range[1]
    c=plot_data.lin_colour_plot(colour_min=colour_min,colour_max=colour_max,ax=ax,cmap=cmap)
    if colourbar:
      if cbar_ax==None:
        c=pl.colorbar(c,ax=ax)
      else:
        c=pl.colorbar(c,cax=cbar_ax)
    fi.plot_save(output,block)
    return c

  def nan_pad(self,time,front=True):
    # WARNING!  This is a hacky fix to a problem in flux-phase diagram creation.  Use at your own risk...
    addbit=self.copy()
    if front:
      addbit.set_x(np.array([self.get_start_time()-time]))
    else:
      addbit.set_x(np.array([self.get_end_time()+time]))
    addbit.set_y(np.array([np.nan]))
    addbit.set_ye(np.array([np.nan]))
    self.add_data(addbit)
    
  def nan_padded(self,time,front=True):
    # WARNING!  This is a hacky fix to a problem in flux-phase diagram creation.  Use at your own risk...
    n=self.copy()
    n.nan_pad(time,front=front)
    return n

  # Phase-folder!  Does a bog-standard fixed period phase-folding

  def plot_with_period(self,period=None,t0=0,output=None,block=False,**kwargs):
    if period==None:
      period=self.fetch_period()
    ax=fi.filter_axes(output)
    self.quickplot(output=ax,**kwargs)
    n0=int((self.get_start_time()-t0)//period)+1
    n1=int((self.get_end_time()-t0)//period)+1
    for n in range(n0,n1):
      ax.axvline(n*period+t0,color='k',zorder=-1)
    fi.plot_save(output,block)

  def get_phases(self,period=None):
    if period==None:
      period=self.fetch_period()
    return (self.get_x()%period)/period

  def get_Ncycles(self,period=None):
    if period==None:
      period=self.fetch_period()
    return self.get_x()/period

  def set_x_axis_to_Ncycles(self,period=None):
    if self.x_axis_is_phase():
      wr.warn('X-axis is already in NCycles!')
    else:
      if period==None:
        period=self.fetch_period()
      self.set_x(self.get_Ncycles(period))
      self.set_xphase_flag(True)
      self.set_t_units('# Cycles')
      self.set_period(period)

  def setted_x_axis_to_Ncycles(self,period=None):  # disregard the awful english, it follows the naming scheme for in-place vs. copy-producing methods...
    if self.x_axis_is_phase():
      wr.warn('X-axis is already in NCycles!')
      return self
    else:
      if period==None:
        period=self.fetch_period()
      s=self.copy()
      s.set_x_axis_to_Ncycles(period)
      return s

  def phase_fold(self,period=None,phase_bins=100,standev_errors=False):
    if self.is_folded():
      wr.warn('Already folded!  Skipping!')
      return None
    if period==None:
      period=self.fetch_period()
    self.folder=fo.linear_folder(self,period)
    self.folder.fold(phase_bins,standev_errors=standev_errors)
    self.set_x(self.folder.get_fx())
    self.set_y(self.folder.get_fy())
    self.set_ye(self.folder.get_fye())
    self.set_meta('folded_period',period)
    self.set_period(period)
    self.set_folded_flag(True)
    self.set_xphase_flag(True)

  def phase_folded(self,period=None,phase_bins=100,standev_errors=False):
    if self.is_folded():
      wr.warn('Already folded!  Skipping!')
      return self
    if period==None:
      period=self.fetch_period()
    s=self.copy()
    s.phase_fold(period,phase_bins,standev_errors=standev_errors)
    return s

  # Varifolder from PANTHEON!  Probably super unstable!

  def varifold(self,zeros,phase_bins=100,standev_errors=False):
    if self.is_folded():
      wr.warn('Already folded!  Skipping!')
    if len(zeros)<2:
      raise dat.DataError('Must provide at least two zero phase points (and preferably quite a few more!)')
    zeros=np.array(zeros)
    zeros.sort()
    self.folder=fo.varifolder(self,zeros)
    self.folder.fold(phase_bins,standev_errors=standev_errors)
    self.set_x(self.folder.get_fx())
    self.set_y(self.folder.get_fy())
    self.set_ye(self.folder.get_fye())
    self.set_meta('phase_zeros',zeros)
    self.set_phase_zeros(zeros)
    self.set_folded_flag(True)
    self.set_xphase_flag(True)
    pl.figure()

  def varifolded(self,zeros,phase_bins=100,standev_errors=False):
    if self.is_folded():
      wr.warn('Already folded!  Skipping!')
      return self
    s=self.copy()
    s.varifold(zeros,phase_bins,standev_errors=standev_errors)
    return s

  # Gets the period at which the dispersion in a folded lightcurve's phase bins is minimum.  One way of working out a period

  def get_minimum_dispersion_period(self,estimate,phase_bins=100,max_iterations=7,variance=0.1,overwrite=True,error=np.inf):
    x=[];y=[]
    def min_dispersion_test_function(p):
      test_lc=self.phase_folded(p,phase_bins=phase_bins,standev_errors=True)
      return np.sum(test_lc.get_ye())
    for i in np.arange(estimate*(1-variance),estimate*(1+variance),estimate*variance/10.):
      x.append(i)
      y.append(min_dispersion_test_function(i))
    x=np.array(x)
    y=np.array(y)
    invals=(np.min(y)-np.max(y),estimate,estimate*0.1*variance,np.max(y))
    try:
      with wr.catch_warnings():
        wr.simplefilter("ignore")
        ests=optm.curve_fit(func.gaussian,x,y,invals)
    except RuntimeError:
      if overwrite or not self.has_period():
        self.set_period(estimate)
      return estimate,error  # return a value if the parameter space becomes non-gaussian
    check0=np.abs(ests[0][0])>3*np.sqrt(np.abs(np.diag(ests[1])))[0]
    check1=np.abs(ests[0][1])>3*np.sqrt(np.abs(np.diag(ests[1])))[1]
    check2=np.abs(ests[0][2])>3*np.sqrt(np.abs(np.diag(ests[1])))[2]
    check3=np.abs(ests[0][3])>3*np.sqrt(np.abs(np.diag(ests[1])))[3]
    if not (check0 and check1 and check2 and check3): # also return value if gaussian is fit but poorly constrained
      if overwrite or not self.has_period():
        self.set_period(estimate)
      return estimate,error
    estimate=ests[0][1]
    error=np.sqrt(np.diag(ests[1])[1])
    if -np.log10(variance)<=max_iterations:
      estimate,error=self.get_minimum_dispersion_period(estimate,phase_bins=phase_bins,max_iterations=max_iterations,variance=0.1*variance,error=error)
    if overwrite or not self.has_period():
      self.set_period(estimate)
    return estimate,error
    
  # Burst properties!  Shameless PANTHEON ripoff!

  def get_burst_properties(self,thresh_hi,thresh_lo,scan_buffer=5,thresh_method='standard deviations',parameter=10,environment_scan=100):
    if thresh_method.lower()=='percentile':
      high_thresh=np.percentile(self.get_y(),thresh_hi)
      low_thresh=np.percentile(self.get_y(),thresh_lo)
    elif thresh_method.lower()=='standard deviations':
      try:
        thresh_method_parameter=float(parameter)
      except:
        raise dat.DataError('Standard Deviation Burst Scanning Requires an input parameter as window size (in time units)') # In units of time
      n_windows=int(self.get_xrange()//parameter)                # find standard deviation in many windows and get median, so bursts dont mess with it
      stds=[]
      st_time=self.get_start_time()
      for i in range(n_windows):
        try:
          scalve=self.calved(st_time+i*parameter,st_time+(i+1)*parameter)
          assert not scalve.is_empty()
          stds.append(scalve.get_std())
        except:
          continue
      useful_std=np.median(stds)
      useful_base=self.get_median()
      high_thresh=useful_base+thresh_hi*useful_std
      low_thresh=useful_base+thresh_lo*useful_std
    else:
      high_thresh=thresh_hi
      low_thresh=thresh_lo
      
    over_thresh=self.get_y()>low_thresh                                   # Create a Boolean array by testing whether the input array is above the mid
                                                                          #  threshold.  Each region of consecutive 'True' objects is considered a burst-
                                                                          #  -candidate region.
    burst_starts=[]
    burst_ends=[]
    bursts=[]
    while True:                                                                                                                          
      masked=np.array(self.get_y())*over_thresh                        # Reduce all data outside of burst-candidate regions to zero
      if max(masked)<high_thresh:                                      # If highest peak in all remaining burst-candidate regions is below the high threshold,
                                                                          #  assume there are no more bursts to be found.
        break
      peak_loc=masked.tolist().index(max(masked))                      # Find peak in remaining data
      i=peak_loc
      end_counter=0
      while i<self.get_length() and end_counter<scan_buffer:                    # Scrub the True objects in the Boolean array corresponding to that peak's candidate
                                                                       #  region, thus removing it
        if not over_thresh[i]:
          end_counter+=1
        else:
          end_counter=0
        over_thresh[i]=False
        i+=1
        end_i=i-scan_buffer
      i=peak_loc-1
      end_counter=0
      while i>=0 and end_counter<scan_buffer:

        if not over_thresh[i]:
          end_counter+=1
        else:
          end_counter=0
        over_thresh[i]=False
        i-=1
        start_i=i+scan_buffer
      if end_i-start_i<scan_buffer:
        continue
      burst_starts.append(start_i)
      burst_ends.append(end_i)
    burst_starts.sort()
    burst_ends.sort()
    num_bursts=len(burst_starts)

    for b in range(num_bursts):
      if b==0:
        s_index=0
      else:
        s_index=burst_ends[b-1]
      if b==num_bursts-1:
        e_index=self.get_length()
      else:
        e_index=burst_starts[b+1]

      if burst_starts[b]-s_index<1:
        preburst_mean=np.nan
      else:
        s_time=self.get_x()[s_index]
        e_time=self.get_x()[burst_starts[b]]
        s_time=max(s_time,e_time-environment_scan)
        preburst_mean=self.calved(s_time,e_time).get_mean()

      if e_index-burst_ends[b]<1:
        postburst_mean=np.nan
      else:
        s_time=self.get_x()[burst_ends[b]]
        e_time=self.get_x()[e_index-1]
        e_time=min(e_time,s_time+environment_scan)
        postburst_mean=self.calved(s_time,e_time).get_mean()
      
      s_time=self.get_x()[burst_starts[b]]
      e_time=self.get_x()[burst_ends[b]]
      burst_lc=self.calved(s_time,e_time)
      if burst_lc.get_length()<scan_buffer:
        continue
      this_burst=burst(burst_lc,preburst_mean,postburst_mean)
      bursts.append(this_burst)

    self.bursts=bursts

  def plot_bursts(self):
    if not self.has('bursts'):
      raise dat.DataError('Bursts not prepared!  Run get_burst_properties first!')
    elif len(self.bursts)==0:
      raise dat.DataError('No Bursts to plot!')
    else:
      for burst in self.bursts:
        burst.lc.quickplot(block=False)

  def show_burst_properties(self):
    if not self.has('bursts'):
      raise dat.DataError('Bursts not prepared!  Run get_burst_properties first!')
    print('Number of Bursts: '+str(len(self.bursts)))
    b_i=0
    for b in self.bursts:
      print('-----------------')
      b_i+=1
      print('Burst #'+str(b_i)+':')
      b.list_parameters()
      
  # Eclipse properties!  Great for those Eclipse Depth/Out-of-Eclipse-Flux plots!

  def get_eclipse_properties(self,period=None,phase=None,sample_halfwidth=0.1,pcm_low=20,pcm_high=95,min_sigma=1):
    if self.is_folded():
      raise dat.DataError('Cannot find eclipse properties of folded data!')
    if not self.x_axis_is_phase():
      if period==None:
        period=self.fetch_period()
        sample_lc=self.setted_x_axis_to_Ncycles(period)
      else:
        sample_lc=self.copy()
    else:
      sample_lc=self.setted_x_axis_to_Ncycles(period)
    smo,det=sample_lc.decomposed((1,pcm_low,pcm_high),'pcm')
    del sample_lc
    if phase==None:
      folded=det.phase_folded(1)
      input_t=folded.get_min_time()
      medi=folded.get_median()
      fit=optm.curve_fit(func.gaussian,folded.get_x(),folded.get_y(),(folded.get_min()-medi,input_t,sample_halfwidth,medi))
      phase=fit[0][1]
    smo.add_time(-phase)
    det.add_time(-phase)
    eclipse_depths=[]
    eclipse_widths=[]
    eclipse_contin=[] # continuum at eclipse, e.g. out-of-eclipse flux

    eclipse_depths_e=[]
    eclipse_widths_e=[]
    eclipse_contin_e=[]

    def fit_function(x,A,sig):
      return func.gaussian_based(x,A,0,sig)

    for t in range(int(smo.get_start_time())+1,int(smo.get_end_time())+1):
      cl_det=det.calved(t-sample_halfwidth,t+sample_halfwidth).added_time(-t)
      if cl_det.is_empty():
        continue
      cl_smo=smo.calved(t-sample_halfwidth,t+sample_halfwidth).added_time(-t)
      try:
        with wr.catch_warnings():
          wr.simplefilter('ignore')
          fit=optm.curve_fit(fit_function,cl_det.get_x(),cl_det.get_y(),(cl_det.get_min(),sample_halfwidth))
      except:
        continue
      fit_v=fit[0]
      with wr.catch_warnings():
        wr.simplefilter('ignore')
        fit_e=np.sqrt(np.diag(fit[1]))
      if not (np.abs(fit_v[0])>min_sigma*fit_e[0] and np.abs(fit_v[1])>min_sigma*fit_e[1]):
        continue
      eclipse_depths.append(fit_v[0])
      eclipse_widths.append(fit_v[1])
      eclipse_contin.append(cl_smo.get_mean())

      eclipse_depths_e.append(fit_e[0])
      eclipse_widths_e.append(fit_e[1])
      eclipse_contin_e.append(cl_smo.get_error_of_mean())

    self.eclipse_depths=-np.array(eclipse_depths)
    self.eclipse_widths=np.array(eclipse_widths)
    self.eclipse_contin=np.array(eclipse_contin)

    self.eclipse_depths_e=np.array(eclipse_depths_e)
    self.eclipse_widths_e=np.array(eclipse_widths_e)
    self.eclipse_contin_e=np.array(eclipse_contin_e)

  def plot_eclipse_depths_continuum_diagram(self,bin_factor=1,output=None,block=False,arrowplot=False,plot1_1=True,q_lim=50,q_lim_absolute=False,star_flux=None,**kwargs):
    if not self.has('eclipse_depths'):
      raise dat.DataError('Must prepare eclipse properties before plotting EDC diagram!')
    raw_x=self.eclipse_contin
    raw_y=self.eclipse_depths
    raw_xe=self.eclipse_contin_e
    raw_ye=self.eclipse_depths_e
    px,py,pxe,pye=dat.rebin_by_factor(bin_factor,raw_x,raw_y,raw_xe,raw_ye)
    ax=fi.filter_axes(output)
    #ax.errorbar(px,py,xerr=pxe,yerr=pye,zorder=-1,**kwargs)
    if arrowplot:
      dat.arrow_plot(ax,px,py,zorder=90)
    if plot1_1:
      if star_flux==None:
        if not q_lim_absolute:
          q_lim=np.percentile(raw_x,q_lim)
          print('Calculated QLim='+str(q_lim))
        fit_x=raw_x[raw_x<q_lim]
        fit_y=raw_y[raw_x<q_lim]
        def fit_func(x,c):
          return x+c
        star_flux=optm.curve_fit(fit_func,fit_x,fit_y,[0])[0][0]
      ax.plot([0.9*min(raw_x),1.1*max(raw_x)],[0.9*min(raw_x)+star_flux,1.1*max(raw_x)+star_flux],'k--',zorder=-1)
    ax.set_xlabel('Out of Eclipse Rate'+self.y_unit_string())
    ax.set_ylabel('Eclipse Depth'+self.y_unit_string())
    pl.show(block=block)

  def plot_flattened_eclipse_depths_continuum_diagram(self,bin_factor=1,output=None,block=False,arrowplot=False,q_lim=50,q_lim_absolute=False,star_flux=None,**kwargs):
    if not self.has('eclipse_depths'):
      raise dat.DataError('Must prepare eclipse properties before plotting EDC diagram!')
    raw_x=self.eclipse_contin
    raw_y=self.eclipse_depths
    raw_xe=self.eclipse_contin_e
    raw_ye=self.eclipse_depths_e
    if star_flux==None:
      def fit_func(x,c):
        return x+c

      if not q_lim_absolute:
        q_lim=np.percentile(raw_x,q_lim)
        print('Calculated QLim='+str(q_lim))
      fit_x=raw_x[raw_x<q_lim]
      fit_y=raw_y[raw_x<q_lim]

      star_flux=optm.curve_fit(fit_func,fit_x,fit_y,[0])[0][0]

    raw_y=raw_y/(raw_x+star_flux)
    raw_ye=(raw_ye)/(raw_x+star_flux)

    px,py,pxe,pye=dat.rebin_by_factor(bin_factor,raw_x,raw_y,raw_xe,raw_ye)
    ax=fi.filter_axes(output)
    ax.errorbar(px,py,xerr=pxe,yerr=pye,color='0.7',zorder=-9000,**kwargs)
    if arrowplot:
      dat.arrow_plot(ax,px,py,zorder=2)
    ax.axhline(1,color='k',zorder=-9000)
    ax.set_xlabel('Out of Eclipse Rate'+self.y_unit_string())
    ax.set_ylabel('Fractional Eclipse Depth')
    pl.show(block=block)

  # Some super basic arithmetic functions for manipulating lightcurves, i.e. "add a constant", "divide by a constant"

  def multiply_by_constant(self,constant):
    self.set_y(self.get_y()*constant)
    self.set_ye(self.get_ye()*constant)

  def multiplied_by_constant(self,constant):
    s=self.copy()
    s.multiply_by_constant(constant)
    return s

  def add_constant(self,constant):
    self.set_y(self.get_y()+constant)

  def added_constant(self,constant):
    s=self.copy()
    s.add_constant(constant)
    return s

  def add_spline(self,spline):
    self.set_y(self.get_y()+spline(self.get_x()))

  def added_spline(self,spline):
    s=self.copy()
    s.add_spline(spline)
    return s

  def divide_by_spline(self,spline,negative=False):
    if negative:
      const=-1
    else:
      const=1
    self.set_y(const*self.get_y()/spline(self.get_x()))
    self.set_ye(self.get_ye()/spline(self.get_x()))

  def divided_by_spline(self,spline):
    s=self.copy()
    s.divide_by_spline(spline)
    return s

  def mask(self,mask):
    if len(mask)!=self.get_length():
      raise dat.DataError('Mask of different length to lightcurve!')
    self.set_x(self.get_x()[mask])
    self.set_y(self.get_y()[mask])
    self.set_ye(self.get_ye()[mask])

  def masked(self,mask):
    s=self.copy()
    s.mask(mask)
    return s

  # making selections on flux

  def flux_cut_between(self,lower,upper):
    if lower>upper:
      raise dat.DataError('Lower flux cut must >= upper flux cut!')
    self.mask(np.logical_and(self.get_y()>=lower,self.get_y()<upper))

  def flux_cutted_between(self,lower,upper):
    s=self.copy()
    s.flux_cut_between(lower,upper)
    return s

  def flux_cut_above(self,limit):
    self.flux_cut_between(limit,np.inf)

  def flux_cutted_above(self,limit):
    s=self.copy()
    s.flux_cut_above(limit)
    return s

  def flux_cut_below(self,limit):
    self.flux_cut_between(-np.inf,limit)

  def flux_cutted_below(self,limit):
    s=self.copy()
    s.flux_cut_below(limit)
    return s

  def clip_percentile_range(self,lower,upper):
    u_bound=np.percentile(self.get_y(),upper)
    l_bound=np.percentile(self.get_y(),lower)
    self.flux_cut_between(self,l_bound,u_bound)

  def clipped_percentile_range(self,lower,upper):
    s=self.copy()
    s.clip_percentile_range(lower,upper)
    return s

  # Some basic statistical properties

  def get_min(self):
    return np.min(self.get_y())

  def get_max(self):
    return np.max(self.get_y())

  def get_mean(self):
    return np.mean(self.get_y())

  def get_error_of_mean(self):
    return np.sqrt(np.sum(self.get_ye()**2))/self.get_length()

  def get_std(self):
    return np.std(self.get_y())

  def get_median(self):
    return np.median(self.get_y())

  def get_range(self):
    return self.get_max()-self.get_min()

class tess_lightcurve(lightcurve):
  def unpack_metadata(self):
    self.objname=self.get_meta('name')
    self.sector=self.get_meta('sector')
    self.mission=self.get_meta('mission')
    self.set_binsize(self.get_meta('binsize'))
    self.set_acceptable_gap(25) # allow a waaaay bigger data gap before a data gap is declared
    abx=np.array(self.get_meta('bx'))
    with wr.catch_warnings():
      wr.filterwarnings('ignore',message='invalid value encountered in less_equal')
      wr.filterwarnings('ignore',message='invalid value encountered in greater_equal')
      if not self.is_empty():
         bgmask=np.logical_and(abx>=self.get_start_time(),abx<=self.get_end_time())
      else:
         bgmask=[]   # Should still be able to initialise a TESS lightcurve of length 0.  This matters for dynamical spectra
    self.set_bx(abx[bgmask].astype(float))  # clipping bg lightcurve to same length as main lc, useful after calving
    self.set_b(np.array(self.get_meta('b'))[bgmask].astype(float))
    self.set_be(np.array(self.get_meta('be'))[bgmask].astype(float))
    self.set_t_units('BJD')
    self.set_y_units('e/s')

class kepler_lightcurve(lightcurve):
  def unpack_metadata(self):
    self.objname=self.get_meta('name')
    self.mission=self.get_meta('mission')
    self.set_binsize(self.get_meta('binsize'))
    self.set_acceptable_gap(100000)
    abx=np.array(self.get_meta('bx'))
    with wr.catch_warnings():
      wr.filterwarnings('ignore',message='invalid value encountered in less_equal')
      wr.filterwarnings('ignore',message='invalid value encountered in greater_equal')
      if not self.is_empty():
         bgmask=np.logical_and(abx>=self.get_start_time(),abx<=self.get_end_time())
      else:
         bgmask=[]   # Should still be able to initialise a Kepler lightcurve of length 0.  This matters for dynamical spectra
    self.set_bx(abx[bgmask].astype(float))  # clipping bg lightcurve to same length as main lc, useful after calving
    self.set_b(np.array(self.get_meta('b'))[bgmask].astype(float))
    self.set_be(np.array(self.get_meta('be'))[bgmask].astype(float))
    self.set_t_units('BJD')
    self.set_y_units('e/s')

class lightcurve_with_gti(lightcurve):

  def shift_gtis(self,shift):
    new_gtis=[]
    for gti in self.gtis:
      new_gtis.append((gti[0]+shift,gti[1]+shift))
    self.gtis=new_gtis


class rxte_lightcurve(lightcurve_with_gti):
  def unpack_metadata(self):
    self.objname=self.get_meta('name')
    self.mission=self.get_meta('mission')
    self.set_binsize(self.get_meta('binsize'))
    self.gtis=self.get_meta('gtis')
    self.min_channel=self.get_meta('min_channel')
    self.max_channel=self.get_meta('max_channel')
    self.set_t_units('s')
    self.set_y_units=('cts/s/PCU')

  def shift_gtis(self,shift):
    new_gtis=[]
    for gti in self.gtis:
      new_gtis.append((gti[0]+shift,gti[1]+shift))
    self.gtis=new_gtis

class nicer_lightcurve(lightcurve_with_gti):
  def unpack_metadata(self):
    self.objname=self.get_meta('name')
    self.mission=self.get_meta('mission')
    self.set_binsize(self.get_meta('binsize'))
    self.gtis=self.get_meta('gtis')
    self.set_t_units('s')
    self.set_y_units=('cts/s')

# ======= bonus classes! ===============================

class burst(object):
  def __init__(self,lc,preburst_mean,postburst_mean):
    self.lc=lc
    self.t_start=self.lc.get_x()[0]
    self.t_end=self.lc.get_x()[-1]
    self.t_peak=self.lc.get_x()[np.argmax(self.lc.get_y())]
    self.preburst_mean=preburst_mean
    self.postburst_mean=postburst_mean
    self.rise_time=self.t_peak-self.t_start
    self.fall_time=self.t_end-self.t_peak

  def plot(self):
    self.lc.quickplot()

  def list_parameters(self):
    print('Start time   : '+str(self.t_start))
    print('End time     : '+str(self.t_end))
    print('Peak time    : '+str(self.t_peak))
    print('Rise time    : '+str(self.rise_time))
    print('Fall time    : '+str(self.fall_time))
    print('Preburst Rate: '+str(self.preburst_mean))
    print('Postburst Rate: '+str(self.postburst_mean))

# ======= define some fetchers for making instrument-specific lightcurves =======================================================================

# THE AUTOFETCH!

def get_lc_auto(filename):
  f=safe_open_fits(filename)
  try:
    mission=f[1].header['TELESCOP'].upper()
  except:
    raise dat.DataError('Unknown Mission!  Cannot load!')
  if mission[:4]=='TESS':
    return get_tess_lc(filename)
  elif mission[:6]=='KEPLER':
    return get_kepler_lc(filename)
  elif mission[:5]=='NICER':
    return get_nicer_lc(filename)
  elif mission[:3]=='XTE':
    mode=f[1].header['DATAMODE']
    if mode=='GoodXenon_2s':
      return get_rxte_lc_from_gx(filename)
    else:
      raise NotImplementedError('RXTE Datamode '+mode+' not implemented!')
  else:
    raise NotImplementedError(mission+' Data not implemented!')

def get_tess_lc(filename):
  if not imported_astropy:
    raise ImportError('No Astropy module found!  Cannot open fits files!')
  imeta={}
  f=safe_open_fits(filename)
  if f[1].header['TELESCOP'][:4].upper()!='TESS':
    raise dat.DataError('FITS file does not appear to be from TESS')
  if f[1].header['EXTNAME'].upper()!='LIGHTCURVE':
    raise dat.DataError('TESS FITS file does not appear to be a lightcurve')

  try:
    assert imported_astroquery
    assert imported_astropy
    radesys=f[0].header['RADESYS'].lower()
    objra=str(f[0].header['RA_OBJ'])+' '
    objdec=str(f[0].header['DEC_OBJ'])
    objcoord=coord.SkyCoord(objra+objdec,frame=radesys,unit='deg')
    with wr.catch_warnings():
      wr.filterwarnings('ignore')
      simbadlist=Simbad.query_region(objcoord, radius='0d1m0s')
    oname=str(simbadlist[0]['MAIN_ID'])[2:-1]
  except (KeyError,TypeError,AssertionError,rx.ConnectionError):
    try:
      oname=f[1].header['OBJECT']
    except KeyError:
      oname='UNKNOWN'
      wr.warn('Could not find Object Name')
  try:
    imeta['tess_id']=f[1].header['OBJECT']
  except:
    imeta['tess_id']='UNKNOWN'
  try:
    sname=str(f[0].header['SECTOR'])
  except:
    sname=''
    wr.warn('Could not find Object Sector')
  imeta['name']=oname
  imeta['sector']=sname
  imeta['mission']='TESS'
  lcdat=f[1].data
  q=lcdat['QUALITY']
  mask=q==0
  x=lcdat['TIME'][mask]
  y=lcdat['SAP_FLUX'][mask]
  ye=lcdat['SAP_FLUX_ERR'][mask]
  imeta['bx']=lcdat['TIME'][mask]
  imeta['b']=lcdat['SAP_BKG'][mask]
  imeta['be']=lcdat['SAP_BKG_ERR'][mask]
  imeta['binsize']=f[1].header['TIMEDEL']
  f.close()
  return tess_lightcurve(x,y,ye,meta=imeta)

###############################################################################

def get_kepler_lc(filename):
  imeta={}
  f=safe_open_fits(filename)
  if f[1].header['TELESCOP'][:6].upper()!='KEPLER':
    raise dat.DataError('FITS file does not appear to be from Kepler')
  if f[1].header['EXTNAME'].upper()!='LIGHTCURVE':
    raise dat.DataError('Kepler FITS file does not appear to be a lightcurve')
  try:
    oname=f[1].header['OBJECT']
  except KeyError:
    oname='UNKNOWN'
    wr.warn('Could not find Object Name')
  try:
    imeta['kepler_id']=f[1].header['KEPLERID']
  except:
    imeta['kepler_id']='UNKNOWN'
  imeta['name']=oname
  imeta['mission']='TESS'
  lcdat=f[1].data
  q=lcdat['SAP_QUALITY']
  mask=q==0
  x=lcdat['TIME'][mask]
  y=lcdat['SAP_FLUX'][mask]
  ye=lcdat['SAP_FLUX_ERR'][mask]
  imeta['bx']=lcdat['TIME'][mask]
  imeta['b']=lcdat['SAP_BKG'][mask]
  imeta['be']=lcdat['SAP_BKG_ERR'][mask]
  imeta['binsize']=f[1].header['TIMEDEL']
  f.close()
  return kepler_lightcurve(x,y,ye,meta=imeta)

###############################################################################

def get_nicer_lc(filename):
  imeta={}
  f=safe_open_fits(filename)
  if f[1].header['TELESCOP'][:5].upper()!='NICER':
    raise dat.DataError('FITS file does not appear to be from NICER')
  if f[1].header['EXTNAME'].upper()!='RATE':
    raise dat.DataError('NICER FITS file does not appear to be a lightcurve')
  try:
    oname=f[1].header['OBJECT']
  except KeyError:
    oname='UNKNOWN'
    wr.warn('Could not find Object Name')
  timezero=f[1].header['TIMEZERO']
  imeta['name']=oname
  imeta['mission']='TESS'
  lcdat=f[1].data
  x=lcdat['TIME']
  y=lcdat['RATE']
  ye=lcdat['ERROR']
  gtimask=np.zeros(len(x),dtype=bool)
  gtis=f[2].data
  for i in range(len(gtis)):
    gtis[i]=(gtis[i][0]-timezero,gtis[i][1]-timezero)
  imeta['gtis']=gtis
  for g in gtis:
    in_gti=np.logical_and(x>=g[0],x<g[1])
    gtimask=np.logical_or(gtimask,in_gti)

  x=x[gtimask]
  y=y[gtimask]
  ye=ye[gtimask]

  imeta['binsize']=f[1].header['TIMEDEL']
  f.close()
  return nicer_lightcurve(x,y,ye,meta=imeta)

###############################################################################

def get_rxte_lc_from_gx(filename,binsize,min_chan=0,max_chan=255):
  imeta={}
  f=safe_open_fits(filename)
  if f[1].header['TELESCOP'][:3].upper()!='XTE':
    raise dat.DataError('FITS file does not appear to be from RXTE')
  if f[1].header['DATAMODE']!='GoodXenon_2s':
    raise dat.DataError('RXTE FITS file does not appear to be a PCA GoodXenon file')
  imeta['mission']='RXTE'
  imeta['name']=f[1].header['OBJECT']
  imeta['binsize']=binsize
  photonlist=np.array(f[1].data['TIME'])
  photon_words=np.array(f[1].data.field(1))

  if min_chan<0:min_chan=0
  if max_chan>255:max_chan=255
  imeta['min_channel']=min_chan
  imeta['max_channel']=max_chan
  if min_chan==0 and max_chan==255:
    pass
  else:
    channels=boolean_words_to_number(photon_words,(16,24))
    photonmask=np.logical_and(channels>=min_chan,channels<=max_chan)
    wr.warning("Filtering by channel?  I'm not 100% sure GX data words are LittleEndian... need to check this later")
    photonlist=photonlist[photonmask]

  t_start=f[1].header['TSTART']
  t_end  =f[1].header['TSTOP']
  t_interval=t_end-t_start
  x=np.arange(t_start,t_end,binsize)
  bin_x=np.arange(0,t_interval,binsize)   # a normalised time used for binning, to prevent float errors
  bin_edges=np.append(bin_x,[t_interval])
  counts=np.histogram(photonlist-t_start,bins=bin_edges)[0] # X axis plus the right-hand edge of the last bin

  y=counts/binsize
  ye=(counts**0.5)/binsize

  gtimask=np.zeros(len(x),dtype=bool)
  gtis=f[2].data
  imeta['gtis']=gtis
  for g in gtis:
    in_gti=np.logical_and(x>=g[0],x<g[1])
    gtimask=np.logical_or(gtimask,in_gti)

  x=x[gtimask]
  y=y[gtimask]
  ye=ye[gtimask]

  return rxte_lightcurve(x,y,ye,meta=imeta)

###############################################################################

def get_lc_from_csv(filename,x_ind=0,y_ind=1,e_ind=2,data_sep=None,meta_sep=':'):
  f=open(filename,'r')
  imeta={}
  x=[]
  y=[]
  ye=[]
  mxind=max(x_ind,y_ind,e_ind)
  for line in f:
    if len(line.split(meta_sep))>1:
      l=line.split(meta_sep)
      if len(l)!=2:
        continue
      mkey=l[0]
      mval=l[1][:-1]
      try:
        mval=float(mval)
        if mval%1==0:
          mval=int(mval)
      except:
        pass
      imeta[mkey]=mval
    elif len(line.split(data_sep))>1:
      l=line.split(data_sep)
      if len(l)<=mxind:
        continue
      try:
        xin=float(l[x_ind])
        yin=float(l[y_ind])
        ein=float(l[e_ind])
      except:
        continue
      x.append(xin)
      y.append(yin)
      ye.append(ein)
  f.close()
  return lightcurve(x,y,ye,meta=imeta)

##############################################################################

def get_lc_from_aavso(filename):
  return get_lc_from_csv(filename,x_ind=0,y_ind=1,e_ind=2,data_sep=',')

##############################################################################

def get_lc_from_arrays(x,y,ye=[]):
  if len(ye)!=len(x):
    ye=np.zeros(len(x))
  return lightcurve(x,y,ye,meta={})

##############################################################################

def get_lc_from_xronos(filename): # convert a xronos output, or similar non-instrumental fits, to a lc object
  f=safe_open_fits(filename)
  imeta={}
  header=f[1].header
  imeta['t_units']=header['TUNIT1']
  imeta['y_units']=header['TUNIT3']
  tstart=header['TSTARTI']+header['TSTARTF']
  imeta['name']=header['OBJECT']
  data=f[1].data
  x=data['TIME']
  y=data['RATE1']
  ye=data['ERROR1']
  return lightcurve(x,y,ye,meta=imeta)

# ======= General purpose lightcurvey gubbins ======

def boolean_words_to_number(words,key,big_endian=False):

  # key is a 2-tuple containing the start and (end+1) indices of the relevant binary passage in the dataword

  words=words[:,key[0]:key[1]]
  decoder=2**np.arange(0,key[1]-key[0],1,dtype=int)
  if big_endian:
    decoder=decoder[::-1]
  decoded_words=np.sum(words*decoder,axis=1)
  return decoded_words

def is_overlap(user_range,test_ranges):
  # user range should be a 2-length iterable, whereas tes_ranges can be a list of 2-length iterables
  assert user_range[1]>user_range[0]
  for test_range in test_ranges:
    assert test_range[1]>test_range[0]
    if np.logical_not(test_range[0]>user_range[1] or user_range[0]>test_range[1]):
      return True
  return False

# ======= Smoothing functions =======

def smart_smooth(lc,parameter,method='None'): # takes a lightcurve-like object
  method=method.lower()
  if method=='none':
    wr.warn('No smoothing method specified: using time mean!')
    return time_mean_smooth(lc,parameter)
  elif method=='savgol':
    return smart_savgol(lc,parameter)
  elif method=='time_median':
    return time_median_smooth(lc,parameter)
  elif method=='time_mean':
    return time_mean_smooth(lc,parameter)
  elif method=='percentile_clipping_time_mean':
    return percentile_clipping_time_mean_smooth(lc,parameter)
  elif method=='pcm':
    return percentile_clipping_time_mean_smooth(lc,parameter)
  elif method=='loess':
    return loess_smooth(lc,parameter)
  else:
    raise NotImplementedError('Unknown smoothing method "'+method+'"!')

def smart_savgol(lc,window_size): # window size in #POINTS units
  y=lc.get_y()
  ws=int(window_size)
  if ws%2==0:
    ws+=1
  with wr.catch_warnings():
    wr.filterwarnings('ignore')
    return sgnl.savgol_filter(y,ws,3)

def time_median_smooth(lc,window_size):  # window size in TIME units
  y=lc.get_y()
  x=lc.get_x()
  ws2=window_size/2.0
  length=len(x)
  newy=np.zeros(length)
  for i in range(length):
    mask=np.logical_and(x>=x[i]-ws2,x<x[i]+ws2)
    newy[i]=np.median(y[mask])
  return newy

def time_mean_smooth(lc,window_size):
  y=lc.get_y()
  x=lc.get_x()
  ws2=window_size/2.0
  length=len(x)
  newy=np.zeros(length)
  for i in range(length):
    mask=np.logical_and(x>=x[i]-ws2,x<x[i]+ws2)
    newy[i]=np.mean(y[mask])
  return newy

def loess_smooth(lc,fraction):
  if not imported_loess:
    raise NotImplementedError
  x=lc.get_x()
  y=lc.get_y()
  loess_data=loe.loess_1d(x,y,frac=fraction)
  return loess_data[1]

def percentile_clipping_time_mean_smooth(lc,in_args):
  try:
    iter(in_args)
    assert len(in_args)==3
  except (AssertionError,TypeError):
    raise TypeError('Input to percentile clipping mean smooth must be iterable of length 3, of the form window size, lower percentile, upper percentile')
  y=lc.get_y()
  x=lc.get_x()
  window_size=in_args[0]
  ws2=window_size/2.0
  lq=in_args[1]
  uq=in_args[2]
  length=len(x)
  newy=np.zeros(length)
  for i in range(length):
    mask1=np.logical_and(x>=x[i]-ws2,x<x[i]+ws2)
    windowed_y=y[mask1]
    lq_val=np.percentile(windowed_y,lq)
    uq_val=np.percentile(windowed_y,uq)
    mask2=np.logical_and(windowed_y>=lq_val,windowed_y<=uq_val)
    newy[i]=np.mean(windowed_y[mask2])
  return newy

# ===== some stats functions =====

def rms(x):
  return (np.mean((x-np.mean(x))**2))**0.5

# ===== a non-OO adding method for lcs =====

def add_lcs(lc_list):
  if not fi.is_iterable(lc_list):
    raise TypeError('add_lcs function must take an list or other iterable containing lc objects!')
  if len(lc_list)==0:
    wr.warn('Empty list passed to add_lcs!  Returning None')
    return None
  base_lc=lc_list[0].copy()
  if not isinstance(base_lc,lightcurve):
    raise TypeError('Element 0 in iterable passed to lc_list is not lc-like!')
  for i in range(1,len(lc_list)):
    addon_lc=lc_list[i].copy()
    if not isinstance(addon_lc,lightcurve):
      raise TypeError('Element '+str(i)+' in iterable passed to lc_list is not lc-like!')
    base_lc.add_data(addon_lc)
  return base_lc

# Utility functions:

#  Wrap the astropy warning with the fits.open call to save adding this warning to eeeeevery loading routine.

def safe_open_fits(filename):
  if not imported_astropy:
    raise ImportError('No Astropy module found!  Cannot open fits files!')
  return fits.open(filename)
