import astropy.coordinates as coord
from   astropy.io import fits
from   astroquery.simbad import Simbad
import freq_lib as frq
from   matplotlib import colors as co
from   matplotlib import pyplot as pl
import numpy as np
import random as rn
from   scipy import fftpack as fou
from   scipy import signal as sgnl
import warnings as wr

# ======== HUMAN READABLE EXCEPTIONS ====================================================================================================================

class DataError(Exception):
  pass

# ========= BASE LIGHTCURVE OBJECT! =====================================================================================================================

class lightcurve(object):

  # Basic initialisation of all lightcurves.  Gets x,y,ye and meta, and attempts to work out units

  def __init__(self,x,y,ye,meta={}):
    tx=np.array(x).astype(float)
    ty=np.array(y).astype(float)
    tye=np.array(ye).astype(float)
    nanmask=np.logical_not(np.logical_and(np.isnan(ty),np.isnan(tye)))
    self.x=tx[nanmask]
    self.y=ty[nanmask]
    self.ye=tye[nanmask]
    self.meta=meta
    self.t_units=''
    self.y_units=''
    self.binsize=0
    self.ft_norm='N/A'
    self.unpack_metadata()
    self.t_unit_string = '' if self.t_units=='' else '('+self.t_units+')'
    self.y_unit_string = '' if self.y_units=='' else '('+self.y_units+')'

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

  # Self-explanatory quick-and-dirty plot machine.  BG plot checks if bg data is available, and dies if not

  def zero_time(self):
    if len(self.x)>0:
       self.x=self.x-self.x[0]
       # Should be able to zero a lightcurve of length 0.  This matters for dynamic spectra

  def quickplot(self,filename=None):
    pl.figure()
    pl.title(self.objname+' Quick Plot')
    pl.errorbar(self.x,self.y,yerr=self.ye)
    pl.xlabel('Time '+self.t_unit_string)
    pl.ylabel('Rate '+self.y_unit_string)
    if filename==None:
      pl.show(block=False)
    else:
      pl.savefig(filename)
      pl.close()

  def quickbgplot(self):
    if 'b' not in self.__dict__:
       raise NotImplementedError('No background data available in '+str(self.__class__)+' object')
    pl.figure()
    pl.title(self.objname+' bg Quick Plot')
    pl.errorbar(self.bx,self.b,yerr=self.be,label='bg')
    pl.errorbar(self.x,self.y,yerr=self.ye,label='phot')
    pl.legend()
    pl.show(block=False)

  # return approx location of significant data gaps (>25* median time separation by default)

  def get_data_gaps(self,min_gap_mult=25):
    delta_T=self.x[1:]-self.x[:-1]
    acceptable_gap=min_gap_mult*np.median(delta_T)
    gap_mask=delta_T>acceptable_gap
    gap_starts=self.x[:-1][gap_mask]
    gap_ends=self.x[1:][gap_mask]
    gaps=[(gap_starts[i],gap_ends[i]) for i in range(len(gap_starts))]
    return gaps

  # Some general Fourier methods

  def fourier(self,norm,normname='custom'):
    wr.warn('Internal Fourier method in lightcurve objects does NOT check for evenly spaced data yet!')
    nyquist=0.5/self.binsize
    if self.binsize==0:
      raise DataError('No binsize provided in metadata!')
    try:
      norm=float(norm)
      custnorm=True
    except ValueError:
      norm=norm.lower()
      custnorm=False
    raw_ft=fou.fft(self.y)
    ft=np.abs(raw_ft[1:len(self.y)//2])**2  # crop all data above nyquist and obtain the amplitudes of the FT
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
    self.ft_freqs=np.linspace(0,nyquist,len(ft)+1)[1:]
      
  def fourier_plot(self):
    if 'ft' not in self.__dict__: self.fourier('leahy')
    pl.figure()
    pl.title(self.objname+' Fourier Spectrum')
    pl.plot(self.ft_freqs,self.ft)
    pl.ylabel('"'+self.ft_norm+'"-normalised power')
    pl.xlabel('Frequency ('+self.t_units+'^-1)')
    pl.show(block=False)

  def fourier_spectrogram(self,binsize,bin_separation):
    raise NotImplementedError('Fourier Spectrogram method not coded yet!')

  # Some general L-S methods

  def lomb_scargle(self,freqrange,norm='auto'):

    # Generalised L-S from Zechmeister & Kuerster, 2009, eq 5-15

    self.ls=frq.lomb_scargle(self.x,self.y,self.ye,freqrange,norm=norm)
    self.ls_freqs=np.array(freqrange)

  def lomb_scargle_plot(self):
    if 'ls' not in self.__dict__: self.lomb_scargle(np.linspace(0,0.05/self.binsize,10000)[1:])
    pl.figure()
    pl.title(self.objname+' Lomb-Scargle Periodogram')
    pl.plot(self.ls_freqs,self.ls)
    pl.ylabel('Lomb-Scargle power')
    pl.xlabel('Frequency ('+self.t_units+'^-1)')
    pl.show(block=False)

  def lomb_scargle_spectrogram(self,freqrange,binsize,bin_separation):
    self.lsparams={'freqrange':freqrange,'binsize':binsize,'bin_separation':bin_separation}
    min_points=80
    progress=0
    starttime=self.x[0]
    numbins=int(((self.x[-1]-starttime)-binsize)//bin_separation)
    lsnorm=(len(self.y)-1)/(2.0*numbins)
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
      calved_lc=self.calve(calve_st,calve_ed)
      calved_lc.lomb_scargle(freqrange,norm=lsnorm)
      lomb_scargle_spec=calved_lc.ls
      dynamic_spectrum[i,:]=lomb_scargle_spec

    self.dynamic_ls_spectrum         = dynamic_spectrum
    self.dynamic_ls_spectrum_tvalues = starttime+bin_separation*(np.arange(0,numbins,1.0))+binsize/2.0
    self.dynamic_ls_spectrum_fvalues = freqrange

  def plot_lomb_scargle_spectrogram(self,colour_range='auto',filename=None):
    if 'dynamic_ls_spectrum' not in self.__dict__:
      wr.warn('Dynamic Lomb-Scargle Spectrum not prepared!  Skipping plotting!')
    else:
      pl.figure(figsize=(20,20))
      pl.title(self.objname+' Dynamic Lomb-Scargle Periodogram')
      pl.xlabel('Time ('+self.t_units+')')
      pl.ylabel('Frequency ('+self.t_units+'^-1)')
      Z=self.dynamic_ls_spectrum.T
      if colour_range=='auto':
        colour_min=np.min(Z[Z>0])
        colour_max=np.max(Z)
      else:
        colour_min=colour_range[0]
        colour_max=colour_range[1]
      c=pl.pcolor(self.dynamic_ls_spectrum_tvalues,self.dynamic_ls_spectrum_fvalues,Z,norm=co.LogNorm(vmin=colour_min, vmax=colour_max))
      pl.colorbar(c)
      if filename==None:
        pl.show(block=False)
      else:
        pl.savefig(filename)
        pl.close()

  # Returns a subset of the lightcurve between t=stime and t=etime.  New lc retains class of parent

  def calve(self,stime,etime):
    with wr.catch_warnings():
      wr.filterwarnings('ignore',message='invalid value encountered in less')
      wr.filterwarnings('ignore',message='invalid value encountered in greater_equal')
      mask=np.logical_and(self.x>=stime,self.x<etime)
    newx=self.x[mask]
    newy=self.y[mask]
    newye=self.ye[mask]
    return self.__class__(newx,newy,newye,self.meta)

  # Attempts to detrend the data for a given window size

  def detrend(self,window_size):
    #ytrend=np.zeros(len(self.y))
    #for i in range(len(self.y)):
    #  window=np.logical_and(self.x>=self.x[i]-time_window/2,self.x<self.x[i]+time_window/2)
    #  ytrend[i]=np.median(self.y[window])
    ws=int(window_size)
    if ws%2==0:
      ws+=1
    with wr.catch_warnings():
      wr.filterwarnings('ignore')
      self.y=self.y-sgnl.savgol_filter(self.y,ws,3)      

  # Shuffler; returns the y,ye pairs in a random order

  def shuffled(self):
    indices=np.arange(len(self.x),dtype=int)
    rn.shuffle(indices)
    newy=self.y[indices]
    newye=self.ye[indices]
    return self.__class__(self.x,newy,newye,self.meta)

class tess_lightcurve(lightcurve):
  def unpack_metadata(self):
    self.objname=self.meta['name']
    self.mission=self.meta['mission']
    self.binsize=self.meta['binsize']
    abx=np.array(self.meta['bx'])
    with wr.catch_warnings():
      wr.filterwarnings('ignore',message='invalid value encountered in less_equal')
      wr.filterwarnings('ignore',message='invalid value encountered in greater_equal')
      if len(self.x)>0:
         bgmask=np.logical_and(abx>=self.x[0],abx<=self.x[-1])
      else:
         bgmask=[]   # Should still be able to initialise a TESS lightcurve of length 0.  This matters for dynamical spectra
    self.bx=abx[bgmask].astype(float)  # clipping bg lightcurve to same length as main lc, useful after calving
    self.b=np.array(self.meta['b'])[bgmask].astype(float)
    self.be=np.array(self.meta['be'])[bgmask].astype(float)
    self.t_units='BJD'
    self.y_units='e/s'

# ======= define some fetchers for making instrument-specific lightcurves =======================================================================

def get_tess_lc(filename):
  imeta={}
  f=fits.open(filename)
  if f[0].header['TELESCOP'][:4].upper()!='TESS':
    raise DataError('FITS file does not appear to be from TESS')
  if f[1].header['EXTNAME'].upper()!='LIGHTCURVE':
    raise DataError('TESS FITS file does not appear to be a lightcurve')

  try:
    radesys=f[0].header['RADESYS'].lower()
    objra=str(f[0].header['RA_OBJ'])+' '
    objdec=str(f[0].header['DEC_OBJ'])
    objcoord=coord.SkyCoord(objra+objdec,frame=radesys,unit='deg')
    with wr.catch_warnings():
      wr.filterwarnings('ignore')
      simbadlist=Simbad.query_region(objcoord, radius='0d1m0s')
    oname=str(simbadlist[0]['MAIN_ID'])[2:-1]
  except (KeyError,TypeError):
    try:
      oname=f[1].header['OBJECT']
    except KeyError:
      oname='UNKNOWN'
      wr.warn('Could not find Object Name')
  try:
    sname=' (Sector '+str(f[0].header['SECTOR'])+')'
  except:
    sname=''
    wr.warn('Could not find Object Name')
  imeta['name']=oname+sname
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

# ======= General purpose lightcurvey gubbins ======

def is_overlap(user_range,test_ranges):
  # user range should be a 2-length iterable, whereas tes_ranges can be a list of 2-length iterables
  assert user_range[1]>user_range[0]
  for test_range in test_ranges:
    assert test_range[1]>test_range[0]
    if np.logical_not(test_range[0]>user_range[1] or user_range[0]>test_range[1]):
      return True
  return False
