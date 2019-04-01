import astropy.coordinates as coord
from   astropy.io import fits
import copy
import data_lib as dat
import file_lib as fi
import freq_lib as frq
from   matplotlib import gridspec as gs
from   matplotlib import pyplot as pl
import numpy as np
import random as rn
from   scipy import fftpack as fou
from   scipy import interpolate as intp
from   scipy import optimize as optm
from   scipy import signal as sgnl
import warnings as wr

# Optional packages

try:
  from astroquery.simbad import Simbad
  imported_astroquery=True
except:
  imported_astroquery=False

try:
  from loess import loess_1d as loe
  imported_loess=True
except:
  imported_loess=False

# ========= BASE LIGHTCURVE OBJECT! =====================================================================================================================

class lightcurve(object):

  # Basic initialisation of all lightcurves.  Gets x,y,ye and meta, and attempts to work out units

  @fi.mjit()
  def __init__(self,x,y,ye,meta={}):
    tx=np.array(x).astype(float)
    ty=np.array(y).astype(float)
    tye=np.array(ye).astype(float)
    nanmask=np.logical_not(np.logical_and(np.isnan(ty),np.isnan(tye)))
    self.x=tx[nanmask]
    self.y=ty[nanmask]
    self.ye=tye[nanmask]
    if 'acceptable_gap' not in meta.keys():  # the gap width before a data gap is declared
      meta['acceptable_gap']=1.5
    self.acceptable_gap=meta['acceptable_gap'] # anything more than 1.5 times the median time separation is considered a data gap
    self.meta=meta
    if 't_units' not in self.__dict__:
      self.t_units=''
    if 'y_units' not in self.__dict__:
      self.y_units=''
    if self.is_empty():
      self.binsize=0
    else:
      self.binsize=min(self.delta_T())
    self.ft_norm='N/A'
    self.unpack_metadata()
    self.is_folded=False
    self.x_axis_is_phase=False
    self.period=0

  # Generic metadata unpacking.  Use this in inherting classes to do fancy stuff with extra data

  def unpack_metadata(self):
    if 'name' in self.meta.keys():
      self.objname=self.meta['name']
    else:
      self.objname='unknown'
    if 'mission' in self.meta.keys():
      self.mission=self.meta['mission']
    else:
      self.mission='unknown'

  # Copy self

  def copy(self):
    return copy.deepcopy(self)

  # Simple data checks

  def is_empty(self):
    return self.get_len()==0

  # Basic getters & setters

  def get_x(self):
    return self.x
  def get_y(self):
    return self.y
  def get_ye(self):
    return self.ye
  def get_acceptable_gap(self):
    return self.acceptable_gap
  def get_xrange(self):
    return self.x[-1]-self.x[0]
  def get_yrange(self):
    return self.y[-1]-self.y[0]
  def get_start_time(self):
    return self.x[0]
  def get_end_time(self):
    return self.x[-1]
  def get_title(self):
    return self.objname
  def get_len(self):
    return len(self.x)

  def set_acceptable_gap(self,gap):
    self.acceptable_gap=gap

  # Get me a spline!

  def get_spline(self,kind='slinear'):
    spline=intp.interp1d(self.x,self.y,kind=kind,fill_value='extrapolate')
    return spline

  # Dump contents to a csv

  def dump(self,filename):
    k=list(self.meta.keys())
    k.sort()
    f=open(filename,'w')
    f.write('META_DATA\n')
    for key in k:
      if type(self.meta[key]) in (str,int,float):
        f.write(key+':'+str(self.meta[key])+'\n')
    f.write('SCIENCE_DATA\n')
    for i in range(self.get_len()):
      f.write(str(self.x[i])+','+str(self.y[i])+','+str(self.ye[i])+'\n')
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
    zeroed_lc=self.copy()
    zeroed_lc.zero_time()
    return zeroed_lc

  def add_time(self,time):
    self.x=self.x+time
    self.shift_gtis(time)
    if 'b' in self.__dict__:
      self.bx=self.bx+time

  def added_time(self,time):
    added_lc=self.copy()
    added_lc.add_time(time)
    return added_lc

  def shift_gtis(self,shift):
    pass    # Placeholder function to allow GTIs to be updated when data is renormed in objects such as RXTE lcs which store this information

  # Self-explanatory quick-and-dirty plot machine.  BG plot checks if bg data is available, and dies if not

  def quickplot(self,output=None,errors=True,block=False,**kwargs):
    if self.is_folded:
      x=np.append(self.x,self.x+1)
      y=np.append(self.y,self.y)
      ye=np.append(self.ye,self.ye)
    else:
      x=self.x
      y=self.y
      ye=self.ye
    ax=fi.filter_axes(output)
    ax.set_title(self.get_title()+' Quick Plot')
    if errors:
      ax.errorbar(x,y,yerr=ye,**kwargs)
    else:
      ax.plot(x,y,**kwargs,**kwargs)
    ax.set_xlabel('Time '+self.t_unit_string())
    ax.set_ylabel('Rate '+self.y_unit_string())
    fi.plot_save(output,block)

  def plot_bg(self,output=None,block=False,**kwargs):
    if 'b' not in self.__dict__:
       raise NotImplementedError('No background data available in '+str(self.__class__)+' object')
    ax=fi.filter_axes(output)
    ax.set_title(self.get_title()+' bg Quick Plot')
    ax.errorbar(self.bx,self.b,yerr=self.be,label='bg',**kwargs)
    ax.errorbar(self.x,self.y,yerr=self.ye,label='phot',**kwargs)
    ax.legend()
    fi.plot_save(output,block)

  # Creates a scatter plot of an unfolded lightcurve where the x-coord of each point is its phase

  def plot_folded_scatterplot(self,period,output=None,block=False,**kwargs):
    if self.is_folded:
      wr.warning("Can't fold that which is already folded!")
    else:
      ax=fi.filter_axes(output)
      p=(self.x%period)/period
      ax.scatter(np.append(p,p+1),np.append(self.y,self.y),marker='.',alpha=min(500/self.get_len(),0.2),color='k',**kwargs)
      ax.set_xlabel('Phase')
      ax.set_ylabel('Rate '+self.y_unit_string())
      fi.plot_save(output,block)

  # return approx location of significant data gaps (>25* median time separation by default)

  def delta_T(self):
    return self.x[1:]-self.x[:-1]

  def get_data_gaps(self):
    delta_T=self.delta_T()
    acceptable_gap=self.acceptable_gap*np.median(delta_T)
    gap_mask=delta_T>acceptable_gap
    gap_starts=self.x[:-1][gap_mask]
    gap_ends=self.x[1:][gap_mask]
    gaps=[(gap_starts[i],gap_ends[i]) for i in range(len(gap_starts))]
    return gaps

  # RMS and associated plots

  def get_rms(self):
    return rms(self.y)

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
    if 'rms_over_time_x' not in self.__dict__:
      raise dat.DataError('RMS plots prepared!  Prepare with prep_variability_stats')
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
    if 'rms_over_time_x' not in self.__dict__:
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
    if 'rms_over_time_x' not in self.__dict__:
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
    if 'rms_over_time_x' not in self.__dict__:
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
    return 0.5/self.binsize

  def fourier(self,norm,normname='custom'):
    wr.warn('Internal Fourier method in lightcurve objects does NOT check for evenly spaced data yet!')
    try:
      norm=float(norm)
      custnorm=True
    except ValueError:
      norm=norm.lower()
      custnorm=False
    raw_ft=fou.fft(self.y)
    ft=np.abs(raw_ft[1:self.get_len()//2])**2  # crop all data above nyquist and obtain the amplitudes of the FT
    if custnorm:
      self.ft_norm=normname
      self.ft=ft*norm
    elif norm=='leahy':
      self.ft_norm='leahy'
      self.ft=frq.leahy(ft,np.sum(self.y))
    elif norm=='rms':
      if 'b' not in self.__dict__: raise NotImplementedError('No background data available to RMS normalise FT in '+str(self.__class__)+' object')
      self.ft_norm='rms'
      self.ft=frq.rms(ft,np.sum(self.y),np.mean(self.y+self.b),np.mean(self.b))
    else:
      if norm!='none':
        wr.warn('Invalid Fourier normalisation '+norm+' specified: using None normalisation')
      self.ft_norm='none'
      self.ft=ft
    self.ft_freqs=np.linspace(0,self.get_nyquist,len(ft)+1)[1:]
      
  def plot_fourier(self,output=None,**kwargs):
    if 'ft' not in self.__dict__: self.fourier('leahy')
    ax=fi.filter_axes(output)
    ax.set_title(self.get_title()+' Fourier Spectrum')
    ax.plot(self.ft_freqs,self.ft,**kwargs)
    ax.set_ylabel('"'+self.ft_norm+'"-normalised power')
    ax.set_xlabel('Frequency ('+self.t_units+'^-1)')
    fi.plot_save(output,**kwargs)

  def fourier_spectrogram(self,binsize,bin_separation):
    raise NotImplementedError('Fourier Spectrogram method not coded yet!')

  # Some general L-S methods

  def lomb_scargle(self,freqrange,norm='auto'):

    # Generalised L-S from Zechmeister & Kuerster, 2009, eq 5-15

    self.ls=frq.lomb_scargle(self.x,self.y,self.ye,freqrange,norm=norm)
    self.ls_freqs=np.array(freqrange)

  def plot_lomb_scargle(self,log=False,output=None,block=False,**kwargs):
    if 'ls' not in self.__dict__: self.lomb_scargle(np.linspace(0,0.05/self.binsize,10000)[1:])
    ax=fi.filter_axes(output)
    ax.set_title(self.get_title()+' Lomb-Scargle Periodogram')
    if log:
      ax.semilogy(self.ls_freqs,self.ls,**kwargs)
    else:
      ax.plot(self.ls_freqs,self.ls,**kwargs)
    ax.set_ylabel('Lomb-Scargle power')
    ax.set_xlabel('Frequency ('+self.t_units+'^-1)')
    fi.plot_save(output,block)

  def get_freq_resolution(self): # get the minimum frequency resolution for LS or Fourier without oversampling
    minf=1/self.get_xrange()
    maxf=self.get_nyquist()
    ndat=self.get_len()
    return (maxf-minf)/ndat

  def fit_qpo(self,f_min,f_max,plot=False,**kwargs):
    frange=f_max-f_min
    newlc=self.copy()
    #newlc.lomb_scargle(np.arange(f_min,f_max,frange/1000))
    newlc.lomb_scargle(np.arange(f_min,f_max,self.get_freq_resolution()))
    newlc.plot_lomb_scargle()
    init_vals=(np.max(newlc.ls_freqs),f_min+frange/2,frange/2,0)
    fit_results=optm.curve_fit(frq.lorentzian,newlc.ls_freqs,newlc.ls,init_vals)
    if plot:
      pl.figure()
      ax=pl.gca()
      newlc.plot_lomb_scargle(output=ax)
      plotrange=np.arange(newlc.ls_freqs[0],newlc.ls_freqs[-1],(newlc.ls_freqs[-1]-newlc.ls_freqs[0])/100)
      ax.plot(plotrange,frq.lorentzian(plotrange,*fit_results[0]),':k',**kwargs)
      pl.show()
    return fit_results

  def lomb_scargle_spectrogram(self,freqrange,binsize,bin_separation):
    min_points=80
    progress=0
    starttime=self.get_start_time()
    numbins=max(int(((self.get_xrange())-binsize)//bin_separation),0)
    if numbins==0:
      raise ValueError('Bin width longer than data set!')
    lsnorm=(self.get_len()-1)/(2.0*numbins)
    dynamic_spectrum=np.zeros((numbins,len(freqrange)))
    data_gaps=self.get_data_gaps()
    for i in range(numbins):
      new_progress=int(i/numbins*100)
      if new_progress>progress:
        print('Lomb_Scargle Periodogram: '+str(progress)+'% complete')
        progress=new_progress
      calve_st=starttime+i*bin_separation
      calve_ed=calve_st+binsize
      if is_overlap((calve_st,calve_ed),data_gaps):
        continue
      calved_lc=self.calved(calve_st,calve_ed)
      calved_lc.lomb_scargle(freqrange,norm=lsnorm)
      lomb_scargle_spec=calved_lc.ls
      dynamic_spectrum[i,:]=lomb_scargle_spec
    dynamic_ls_spectrum         = dynamic_spectrum.T
    dynamic_ls_spectrum_tvalues = starttime+bin_separation*(np.arange(0,numbins,1.0))+binsize/2.0
    dynamic_ls_spectrum_fvalues = freqrange
    dmeta={'x_dimension':'Time','y_dimension':'Frequency','x_unit':self.t_units,'y_unit':self.y_units}
    dls_data=dat.TwoD_Dataframe(dynamic_ls_spectrum_tvalues,dynamic_ls_spectrum_fvalues,dynamic_ls_spectrum,meta=dmeta)
    self.dynamic_ls_data=dls_data

  def dump_lomb_scargle_spectrogram(self,filename,header=True):
    if 'dynamic_ls_data' not in self.__dict__:
      wr.warn('Dynamic Lomb-Scargle Spectrum not prepared!  Skipping csv dump!')
      return None
    self.dynamic_ls_data.dump(filename,header=header)

  def get_lomb_scargle_spectrogram(self):
    if 'dynamic_ls_data' not in self.__dict__:
      wr.warn('Dynamic Lomb-Scargle Spectrum not prepared!  Skipping grabbing!')
    return self.dynamic_ls_data

  def plot_lomb_scargle_spectrogram(self,colour_range='auto',filename=None,with_lc=True,with_1d_ls=True,block=False):

    if 'dynamic_ls_data' not in self.__dict__:
      wr.warn('Dynamic Lomb-Scargle Spectrum not prepared!  Skipping plotting!')
      return None
    elif self.dynamic_ls_data.get_max()<=0:
      wr.warn('Dynamic LS spectrum is empty!  Is your Acceptable Gap parameter too small?')
      return None
    if with_lc:
      self.lomb_scargle(self.dynamic_ls_data.get_y())  # if the user wants a 1D LS plotted but hasnt made one yet, make one with the same params as the 2D LS

    size_ratio=4
    grid=gs.GridSpec(size_ratio+with_lc,size_ratio+with_1d_ls)

    fig=pl.figure(figsize=(20,20))
    ax_main=fig.add_subplot(grid[:size_ratio,:size_ratio])
    ax_main.set_title(self.get_title()+' Dynamic Lomb-Scargle Periodogram')
    if not with_lc:
      ax_main.set_xlabel('Time '+self.t_unit_string())
    ax_main.set_ylabel('Frequency ('+self.t_units+'^-1)')
    Z=self.dynamic_ls_data.get_z()
    if colour_range=='auto':
      colour_min=np.min(Z[Z>0])
      colour_max=np.max(Z)
    else:
      colour_min=colour_range[0]
      colour_max=colour_range[1]
    c=self.dynamic_ls_data.log_colour_plot(colour_min=colour_min,colour_max=colour_max,ax=ax_main)
    if (not with_lc) and (not with_1d_ls):
      pl.colorbar(c,ax=ax_main)
    if with_lc:
      ax_main.set_xticks([])
      ax_lc=fig.add_subplot(grid[size_ratio,:size_ratio])
      self.quickplot(output=ax_lc)
      ax_lc.set_ylabel('Rate '+self.y_unit_string())
      ax_lc.set_xlim(min(self.dynamic_ls_data.get_x()),max(self.dynamic_ls_data.get_x()))
      ax_lc.set_xlabel('Time '+self.t_unit_string())
    if with_1d_ls:
      ax_ls=fig.add_subplot(grid[:size_ratio,size_ratio])
      ax_ls.semilogx(self.ls,self.ls_freqs,'k')
      ax_ls.set_yticks([])
      ax_ls.set_xlabel('L-S Power')
      ax_ls.fill_betweenx(self.ls_freqs,self.ls,0,facecolor='0.7')
      ax_ls.set_xlim(np.percentile(self.ls,25),max(self.ls*1.01))
      ax_ls.set_ylim(min(self.dynamic_ls_data.get_y()),max(self.dynamic_ls_data.get_y()))
    fi.plot_save(filename,block)

  # Add data from a matching lc object

  def add_data(self,lc):
    if lc.__class__!=self.__class__:
      raise TypeError('Cannot concatenate '+str(lc.__class__)+' and '+str(self.__class__)+'!')
    dictkeys1=self.__dict__
    dictkeys2=lc.__dict__
    keyset=(set(dictkeys1)|set(dictkeys2))-set(('x','y','ye','bx','b','be','meta','sector')) #some protected dict items which are allowed to mismatch
    for key in keyset:
      if key not in self.__dict__:
        self.__dict__[key]=lc.__dict__[key]
      elif key not in lc.__dict__:
        pass
      elif self.__dict__[key]!=lc.__dict__[key]:
        wr.warn('Warning: '+key.upper()+' does not match!')
    if ('b' in self.__dict__) and ('b' in lc.__dict__):
      both_bg=True
    else:
      both_bg=False
    if lc.get_start_time()>self.get_end_time():
      self.x=np.append(self.x,lc.x)
      self.y=np.append(self.y,lc.y)
      self.ye=np.append(self.ye,lc.ye)
      if both_bg:
        self.bx=np.append(self.bx,lc.bx)
        self.b=np.append(self.b,lc.b)
        self.be=np.append(self.be,lc.be)
    elif lc.get_end_time()<self.get_start_time():
      self.x=np.append(lc.x,self.x)
      self.y=np.append(lc.y,self.y)
      self.ye=np.append(lc.ye,self.ye)
      if both_bg:
        self.bx=np.append(lc.bx,self.bx)
        self.b=np.append(lc.b,self.b)
        self.be=np.append(lc.be,self.be)
    else:
      raise dat.DataError('Datasets overlap in time!  Cannot concatenate!')
    if 'sector' in self.__dict__:
      self.sector=''  # nuke sector info if several sectors have been cated
      self.meta['sector']=''

  def added_data(self,lc):
    lc0=self.copy()
    lc0.add_data(lc)
    return lc0

  # Plot label generation

  def t_unit_string(self):
    if self.t_units=='':
      return ''
    else:
      return '('+self.t_units+')'

  def y_unit_string(self):
    if self.y_units=='':
      return ''
    else:
      return '('+self.y_units+')'

  #### These functions have an in-place version, and a -ed version which returns a new object ####

  # Returns a subset of the lightcurve between t=stime and t=etime.  New lc retains class of parent

  def calve(self,stime,etime):
    with wr.catch_warnings():
      wr.filterwarnings('ignore',message='invalid value encountered in less')
      wr.filterwarnings('ignore',message='invalid value encountered in greater_equal')
      mask=np.logical_and(self.x>=stime,self.x<etime)
    self.x=self.x[mask]
    self.y=self.y[mask]
    self.ye=self.ye[mask]

  def calved(self,stime,etime):
    calved_lc=self.copy()
    calved_lc.calve(stime,etime)
    return calved_lc

  # Attempts to detrend the data for a given window size.

  def detrend(self,window_size,method='savgol'):
    newy=self.y-smart_smooth(self,window_size,method)
    self.y=newy

  def detrended(self,window_size,method='savgol'):
    detrended_lc=self.copy()
    detrended_lc.detrend(window_size,method)
    return detrended_lc

  def plot_with_trend(self,window_size,method='savgol',output=None,block=False,**kwargs):
    ax=fi.filter_axes(output)
    smoothed_lc=self.smoothed(window_size,method)
    self.quickplot(output=ax,**kwargs)
    smoothed_lc.quickplot(output=ax,errors=False,**kwargs)
    fi.plot_save(output,block)

  # smooths the data for a given window size.  Sorta the opposite of above.

  def smooth(self,window_size,method='savgol'):
    newy=smart_smooth(self,window_size,method)
    self.y=newy

  def smoothed(self,window_size,method='savgol'):
    smoothed_lc=self.copy()
    smoothed_lc.smooth(window_size,method)
    return smoothed_lc

  # Shuffler; returns the y,ye pairs in a random order

  def shuffle(self):
    indices=np.arange(self.get_len(),dtype=int)
    rn.shuffle(indices)
    self.y=self.y[indices]
    self.ye=self.ye[indices]

  def shuffled(self):
    shuffled_lc=self.copy()
    shuffled_lc.shuffle()
    return shuffled_lc

  # Phase-folder!  Does a bog-standard fixed period phase-folding

  def get_phases(self,period):
    return (self.x%period)/period

  def get_Ncycles(self,period):
    return self.x/period

  def set_x_axis_to_Ncycles(self,period):
    if self.x_axis_is_phase:
      wr.warn('X-axis is already in NCycles!')
    else:
      self.x=self.get_Ncycles(period)
      self.x_axis_is_phase=True
      self.t_units='# Cycles'
      self.period=period

  def setted_x_axis_to_Ncycles(self,period):  # disregard the awful english, it follows the naming scheme for in-place vs. copy-producing methods...
    if self.x_axis_is_phase:
      wr.warn('X-axis is already in NCycles!')
    else:
      Ncycles_lc=self.copy()
      Ncycles_lc.set_x_axis_to_Ncycles(period)
      return Ncycles_lc

  def phase_fold(self,period,phase_bins=100):
    folder=phase_folder(self,period,phase_bins)
    self.x=folder.get_x()
    self.y=folder.get_y()
    self.ye=folder.get_ye()
    self.meta['folded_period']=period
    self.is_folded=True
    self.period=period
    self.x_axis_is_phase=True

  def phase_folded(self,period,phase_bins=100):
    folded_lc=self.copy()
    folded_lc.phase_fold(period,phase_bins)
    return folded_lc

  # Gets the period at which the dispersion in a folded lightcurve's phase bins is minimum.  One way of working out a period

  def get_minimum_dispersion_period(self,estimate,phase_bins=100,max_iterations=7,variance=0.1,error=np.inf):
    x=[];y=[]
    def min_dispersion_test_function(p):
      test_lc=self.phase_folded(p,phase_bins=phase_bins)
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
        ests=optm.curve_fit(frq.gaussian,x,y,invals)
    except RuntimeError:
      return estimate,error  # return a value if the parameter space becomes non-gaussian
    check0=np.abs(ests[0][0])>3*np.sqrt(np.abs(np.diag(ests[1])))[0]
    check1=np.abs(ests[0][1])>3*np.sqrt(np.abs(np.diag(ests[1])))[1]
    check2=np.abs(ests[0][2])>3*np.sqrt(np.abs(np.diag(ests[1])))[2]
    check3=np.abs(ests[0][3])>3*np.sqrt(np.abs(np.diag(ests[1])))[3]
    if not (check0 and check1 and check2 and check3): # also return value if gaussian is fit but poorly constrained
      return estimate,error
    estimate=ests[0][1]
    error=np.sqrt(np.diag(ests[1])[1])
    if -np.log10(variance)<=max_iterations:
      estimate,error=self.get_minimum_dispersion_period(estimate,phase_bins=phase_bins,max_iterations=max_iterations,variance=0.1*variance,error=error)
    return estimate,error
    
    

  # Some super basic arithmetic functions for manipulating lightcurves, i.e. "add a constant", "divide by a constant"

  def multiply_by_constant(self,constant):
    self.y=self.y*constant
    self.ye=self.ye*constant

  def multiplied_by_constant(self,constant):
    multiplied_lc=self.copy()
    multiplied_lc.multiply_by_constant(constant)
    return multiplied_lc

  def add_constant(self,constant):
    self.y=self.y+constant

  def added_constant(self,constant):
    added_lc=self.copy()
    added_lc.add_constant(constant)
    return added_lc

  def add_spline(self,spline):
    self.y=self.y+spline(self.x)

  def added_spline(self,spline):
    added_lc=self.copy()
    added_lc.add_spline(spline)
    return added_lc

  def divide_by_spline(self,spline):
    self.y=self.y/-spline(self.x)
    self.ye=self.ye/spline(self.x)

  def divided_by_spline(self,spline):
    divided_lc=self.copy()
    divided_lc.divide_by_spline(spline)
    return divided_lc

  def mask(self,mask):
    if len(mask)!=self.get_len():
      raise dat.DataError('Mask of different length to lightcurve!')
    self.x=self.x[mask]
    self.y=self.y[mask]
    self.ye=self.ye[mask]

  def masked(self,mask):
    masked_lc=self.copy()
    masked_lc.mask(mask)
    return masked_lc

  def clip_percentile_range(self,lower,upper):
    if lower>upper:
      raise dat.DataError('Upper percentile must >= lower percentile!')
    u_bound=np.percentile(self.y,upper)
    l_bound=np.percentile(self.y,lower)
    self.mask(np.logical_and(self.y>=l_bound,self.y<u_bound))

  def clipped_percentile_range(self,lower,upper):
    clipped_lc=self.copy()
    clipped_lc.clip_percentile_range(lower,upper)
    return clipped_lc

  # Some basic statistical properties

  def get_min(self):
    return np.min(self.y)

  def get_max(self):
    return np.max(self.y)

  def get_mean(self):
    return np.mean(self.y)

  def get_std(self):
    return np.std(self.y)

  def get_median(self):
    return np.median(self.y)

  def get_range(self):
    return self.get_max()-self.get_min()

class tess_lightcurve(lightcurve):
  def unpack_metadata(self):
    self.objname=self.meta['name']
    self.sector=self.meta['sector']
    self.mission=self.meta['mission']
    self.binsize=self.meta['binsize']
    self.acceptable_gap=25 # allow a waaaay bigger data gap before a data gap is declared
    abx=np.array(self.meta['bx'])
    with wr.catch_warnings():
      wr.filterwarnings('ignore',message='invalid value encountered in less_equal')
      wr.filterwarnings('ignore',message='invalid value encountered in greater_equal')
      if not self.is_empty():
         bgmask=np.logical_and(abx>=self.get_start_time(),abx<=self.get_end_time())
      else:
         bgmask=[]   # Should still be able to initialise a TESS lightcurve of length 0.  This matters for dynamical spectra
    self.bx=abx[bgmask].astype(float)  # clipping bg lightcurve to same length as main lc, useful after calving
    self.b=np.array(self.meta['b'])[bgmask].astype(float)
    self.be=np.array(self.meta['be'])[bgmask].astype(float)
    self.t_units='BJD'
    self.y_units='e/s'

  def get_title(self): # so that the sector is loaded onto plot titles by default
    if self.sector=='':
      return self.objname
    return self.objname+' (Sector '+self.sector+')'

class rxte_lightcurve(lightcurve):
  def unpack_metadata(self):
    self.objname=self.meta['name']
    self.mission=self.meta['mission']
    self.binsize=self.meta['binsize']
    self.gtis=self.meta['gtis']
    self.min_channel=self.meta['min_channel']
    self.max_channel=self.meta['max_channel']
    self.t_units='s'
    self.y_units='cts/s/PCU'

  def shift_gtis(self,shift):
    new_gtis=[]
    for gti in self.gtis:
      new_gtis.append((gti[0]+shift,gti[1]+shift))
    self.gtis=new_gtis

# ======= define some fetchers for making instrument-specific lightcurves =======================================================================

def get_tess_lc(filename):
  imeta={}
  f=fits.open(filename)
  if f[1].header['TELESCOP'][:4].upper()!='TESS':
    raise dat.DataError('FITS file does not appear to be from TESS')
  if f[1].header['EXTNAME'].upper()!='LIGHTCURVE':
    raise dat.DataError('TESS FITS file does not appear to be a lightcurve')

  try:
    assert imported_astroquery
    radesys=f[0].header['RADESYS'].lower()
    objra=str(f[0].header['RA_OBJ'])+' '
    objdec=str(f[0].header['DEC_OBJ'])
    objcoord=coord.SkyCoord(objra+objdec,frame=radesys,unit='deg')
    with wr.catch_warnings():
      wr.filterwarnings('ignore')
      simbadlist=Simbad.query_region(objcoord, radius='0d1m0s')
    oname=str(simbadlist[0]['MAIN_ID'])[2:-1]
  except (KeyError,TypeError,AssertionError):
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

def get_rxte_lc_from_gx(filename,binsize,min_chan=0,max_chan=255):
  imeta={}
  f=fits.open(filename)
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

def get_lc_from_csv(filename,x_ind=0,y_ind=1,e_ind=2,data_sep=',',meta_sep=':'):
  f=open(filename,'r')
  imeta={}
  x=[]
  y=[]
  ye=[]
  mxind=max(x_ind,y_ind,e_ind)
  for line in f:
    if meta_sep in line:
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
    elif data_sep in line:
      l=line.split(data_sep)
      if len(l)<=mxind:
        continue
      x.append(float(l[x_ind]))
      y.append(float(l[y_ind]))
      ye.append(float(l[e_ind]))
  f.close()
  return lightcurve(x,y,ye,meta=imeta)

##############################################################################

def get_lc_from_arrays(x,y,ye=[]):
  if len(ye)!=len(x):
    ye=np.zeros(len(x))
  return lightcurve(x,y,ye,meta={})

##############################################################################

def get_lc_from_xronos(filename): # convert a xronos output, or similar non-instrumental fits, to a lc object
  f=fits.open(filename)
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

def smart_smooth(lc,parameter,method): # takes a lightcurve-like object
  method=method.lower()
  if method=='savgol':
    return smart_savgol(lc,parameter)
  elif method=='time_median':
    return time_median_smooth(lc,parameter)
  elif method=='time_mean':
    return time_mean_smooth(lc,parameter)
  elif method=='percentile_clipping_time_mean':
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

# ====== Folding! ======

class phase_folder(object):
  def __init__(self,lc,period,phase_bins,standev_errors=True):    
    # Two different errors can be returned.  Default is the standard deviation of the distribution of y in each bin.  Otherwise can just propagate errors normally.
    x=lc.get_x()
    y=lc.get_y()
    ye=lc.get_ye()
    px=(x%period)/period
    py=np.zeros(phase_bins)
    pye=np.zeros(phase_bins)
    for i in range(phase_bins):
      lower_phase=i/phase_bins
      upper_phase=(i+1)/phase_bins
      mask=np.logical_and(px>=lower_phase,px<upper_phase)
      py[i]=np.mean(y[mask])
      if standev_errors:
        pye[i]=np.std(y[mask])
      else:
        pye[i]=np.sqrt(np.sum(ye[mask]**2)/np.sum(mask))
    self.x=np.arange(0,1,1/phase_bins)
    self.y=py
    self.ye=pye
    self.period=period
    self.phase_bins=phase_bins

  def get_x(self):
    return self.x
  def get_y(self):
    return self.y
  def get_ye(self):
    return self.ye

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
