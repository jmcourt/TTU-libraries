import astropy.coordinates as coord
from   astropy.io import fits
from   astroquery.simbad import Simbad
import data_lib as dat
import file_lib as fi
import freq_lib as frq
from   matplotlib import gridspec as gs
from   matplotlib import pyplot as pl
import numpy as np
import random as rn
from   scipy import fftpack as fou
from   scipy import signal as sgnl
import warnings as wr

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
    if 'acceptable_gap' not in meta.keys():  # the gap width before a data gap is declared
      meta['acceptable_gap']=1.5
    self.acceptable_gap=meta['acceptable_gap'] # anything more than 1.5 times the median time separation is considered a data gap
    self.meta=meta
    self.t_units=''
    self.y_units=''
    self.binsize=min(self.delta_T())
    self.ft_norm='N/A'
    self.unpack_metadata()
    self.folded=False
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

  # Getters

  def get_x(self):
    return self.x
  def get_y(self):
    return self.y
  def get_ye(self):
    return self.ye
  def get_acceptable_gap(self):
    return self.acceptable_gap
  def set_acceptable_gap(self,gap):
    self.acceptable_gap=gap

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
    for i in range(len(self.x)):
      f.write(str(self.x[i])+','+str(self.y[i])+','+str(self.ye[i])+'\n')
    f.close()

  # Self-explanatory quick-and-dirty plot machine.  BG plot checks if bg data is available, and dies if not

  def zero_time(self):
    if len(self.x)>0:
       self.x=self.x-self.x[0]
       # Should be able to zero a lightcurve of length 0.  This matters for dynamic spectra

  def quickplot(self,filename=None):
    if self.folded:
      x=np.append(self.x,self.x+1)
      y=np.append(self.y,self.y)
      ye=np.append(self.ye,self.ye)
    else:
      x=self.x
      y=self.y
      ye=self.ye
    pl.figure()
    pl.title(self.objname+' Quick Plot')
    pl.errorbar(x,y,yerr=ye)
    pl.xlabel('Time '+self.t_unit_string)
    pl.ylabel('Rate '+self.y_unit_string)
    fi.plot_save(filename)

  def plot_bg(self,filename=None):
    if 'b' not in self.__dict__:
       raise NotImplementedError('No background data available in '+str(self.__class__)+' object')
    pl.figure()
    pl.title(self.objname+' bg Quick Plot')
    pl.errorbar(self.bx,self.b,yerr=self.be,label='bg')
    pl.errorbar(self.x,self.y,yerr=self.ye,label='phot')
    pl.legend()
    fi.plot_save(filename)

  # Creates a scatter plot of an unfolded lightcurve where the x-coord of each point is its phase

  def plot_folded_scatterplot(self,period,filename=None):
    if self.folded:
      wr.warning("Can't fold data which is already folded!")
    else:
      p=(self.x%period)/period
      pl.scatter(np.append(p,p+1),np.append(self.y,self.y),marker='.',alpha=0.01,color='k')
      fi.plot_save(filename)

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

  # Some general Fourier methods

  def fourier(self,norm,normname='custom'):
    wr.warn('Internal Fourier method in lightcurve objects does NOT check for evenly spaced data yet!')
    nyquist=0.5/self.binsize
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
      
  def plot_fourier(self,filename=None):
    if 'ft' not in self.__dict__: self.fourier('leahy')
    pl.figure()
    pl.title(self.objname+' Fourier Spectrum')
    pl.plot(self.ft_freqs,self.ft)
    pl.ylabel('"'+self.ft_norm+'"-normalised power')
    pl.xlabel('Frequency ('+self.t_units+'^-1)')
    fi.plot_save(filename)

  def fourier_spectrogram(self,binsize,bin_separation):
    raise NotImplementedError('Fourier Spectrogram method not coded yet!')

  # Some general L-S methods

  def lomb_scargle(self,freqrange,norm='auto'):

    # Generalised L-S from Zechmeister & Kuerster, 2009, eq 5-15

    self.ls=frq.lomb_scargle(self.x,self.y,self.ye,freqrange,norm=norm)
    self.ls_freqs=np.array(freqrange)

  def plot_lomb_scargle(self,log=False,filename=None):
    if 'ls' not in self.__dict__: self.lomb_scargle(np.linspace(0,0.05/self.binsize,10000)[1:])
    pl.figure()
    pl.title(self.objname+' Lomb-Scargle Periodogram')
    if log:
      pl.semilogy(self.ls_freqs,self.ls)
    else:
      pl.plot(self.ls_freqs,self.ls)
    pl.ylabel('Lomb-Scargle power')
    pl.xlabel('Frequency ('+self.t_units+'^-1)')
    fi.plot_save(filename)

  def lomb_scargle_spectrogram(self,freqrange,binsize,bin_separation):
    self.lsparams={'freqrange':freqrange,'binsize':binsize,'bin_separation':bin_separation}
    min_points=80
    progress=0
    starttime=self.x[0]
    numbins=max(int(((self.x[-1]-starttime)-binsize)//bin_separation),0)
    if numbins==0:
      raise ValueError('Bin width longer than data set!')
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
      calved_lc=self.calved(calve_st,calve_ed)
      calved_lc.lomb_scargle(freqrange,norm=lsnorm)
      lomb_scargle_spec=calved_lc.ls
      dynamic_spectrum[i,:]=lomb_scargle_spec

#### dynamic_ls_spectrum, dynamic_ls_spectrum_tvalues, dynamic_ls_spectrum_fvalues

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

  def plot_lomb_scargle_spectrogram(self,colour_range='auto',filename=None,with_lc=True,with_1d_ls=True):

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
    ax_main.set_title(self.objname+' Dynamic Lomb-Scargle Periodogram')
    if not with_lc:
      ax_main.set_xlabel('Time ('+self.t_units+')')
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
      ax_lc.plot(self.x,self.y,'k')
      ax_lc.set_ylabel('Rate '+self.y_unit_string)
      ax_lc.set_xlim(min(self.dynamic_ls_data.get_x()),max(self.dynamic_ls_data.get_x()))
      ax_lc.set_xlabel('Time ('+self.t_units+')')
    if with_1d_ls:
      ax_ls=fig.add_subplot(grid[:size_ratio,size_ratio])
      ax_ls.semilogx(self.ls,self.ls_freqs,'k')
      ax_ls.set_yticks([])
      ax_ls.set_xlabel('L-S Power')
      ax_ls.fill_betweenx(self.ls_freqs,self.ls,0,facecolor='0.7')
      ax_ls.set_xlim(np.percentile(self.ls,25),max(self.ls*1.01))
      ax_ls.set_ylim(min(self.dynamic_ls_data.get_y()),max(self.dynamic_ls_data.get_y()))
    fi.plot_save(filename)

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
    with wr.catch_warnings():
      wr.filterwarnings('ignore',message='invalid value encountered in less')
      wr.filterwarnings('ignore',message='invalid value encountered in greater_equal')
      mask=np.logical_and(self.x>=stime,self.x<etime)
    newx=self.x[mask]
    newy=self.y[mask]
    newye=self.ye[mask]
    return self.__class__(newx,newy,newye,self.meta)

  # Attempts to detrend the data for a given window size.

  def detrend(self,window_size,method='savgol'):
    newy=self.y-smart_smooth(self,window_size,method)
    self.y=newy

  def detrended(self,window_size,method='savgol'):
    newy=self.y-smart_smooth(self,window_size,method)
    return self.__class__(self.x,newy,self.y,self.meta)

  # smooths the data for a given window size.  Sorta the opposite of above.

  def smooth(self,window_size,method='savgol'):
    newy=smart_smooth(self,window_size,method)
    self.y=newy

  def smoothed(self,window_size,method='savgol'):
    newy=smart_smooth(self,window_size,method)
    return self.__class__(self.x,newy,self.y,self.meta)

  # Shuffler; returns the y,ye pairs in a random order

  def shuffle(self):
    indices=np.arange(len(self.x),dtype=int)
    rn.shuffle(indices)
    self.y=self.y[indices]
    self.ye=self.ye[indices]

  def shuffled(self):
    indices=np.arange(len(self.x),dtype=int)
    rn.shuffle(indices)
    newy=self.y[indices]
    newye=self.ye[indices]
    return self.__class__(self.x,newy,newye,self.meta)

  # Phase-folder!  Does a bog-standard fixed period phase-folding

  def phase_fold(self,period,phase_bins=100):
    folder=phase_folder(self,period,phase_bins)
    self.x=folder.get_x()
    self.y=folder.get_y()
    self.ye=folder.get_ye()
    self.meta['folded_period']=period
    self.folded=True

  def phase_folded(self,period,phase_bins=100):
    folder=phase_folder(self,period,phase_bins)
    newx=folder.get_x()
    newy=folder.get_y()
    newye=folder.get_ye()
    newmeta=self.meta
    newmeta['folded_period']=period
    newobj=self.__class__(newx,newy,newye,newmeta)
    newobj.folded=True
    return newobj

  # Some super basic arithmetic functions for manipulating lightcurves, i.e. "add a constant", "divide by a constant"

  def multiply_by_constant(self,constant):
    self.y=self.y*constant
    self.ye=self.ye*constant

  def multiplied_by_constant(self,constant):
    newy=self.y*constant
    newye=self.ye*constant
    return self.__class__(self.x,newy,newye,self.meta)

  def add_constant(self,constant):
    self.y=self.y+constant

  def added_constant(self,constant):
    newy=self.y+constant
    return self.__class__(self.x,newy,self.ye,self.meta)

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

class tess_lightcurve(lightcurve):
  def unpack_metadata(self):
    self.objname=self.meta['name']
    self.mission=self.meta['mission']
    self.binsize=self.meta['binsize']
    self.acceptable_gap=25 # allow a waaaay bigger data gap before a data gap is declared
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

# ======= define some fetchers for making instrument-specific lightcurves =======================================================================

def get_tess_lc(filename):
  imeta={}
  f=fits.open(filename)
  if f[1].header['TELESCOP'][:4].upper()!='TESS':
    raise dat.DataError('FITS file does not appear to be from TESS')
  if f[1].header['EXTNAME'].upper()!='LIGHTCURVE':
    raise dat.DataError('TESS FITS file does not appear to be a lightcurve')

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
    imeta['tess_id']=f[1].header['OBJECT']
  except:
    imeta['tess_id']='UNKNOWN'
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
  photonlist=np.array([f[1].data['TIME']])
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
  testx_s=x[:,np.newaxis]
  testx_e=testx_s+binsize
  if binsize<0.1:
    wr.warn('The way I pack photons in bins tends to memerror for small bin sizes.  Maybe play with np.hist?')
  boolean_less=np.less(photonlist,testx_e)
  boolean_greq=np.greater_equal(photonlist,testx_s)
  boolean_master=np.logical_and(boolean_less,boolean_greq) # a 2D array, #photons by #bins, with a True or False indicating whether than photon should be in that bin
  counts=np.sum(boolean_master,axis=1) # number of photons in each bin
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

def smart_smooth(lc,window_size,method): # takes a lightcurve-like object
  method=method.lower()
  if method=='savgol':
    return smart_savgol(lc,window_size)
  elif method=='time_median':
    return time_median_smooth(lc,window_size)
  else:
    raise NotImplementedError('Unknown smoothing method "'+method+'"!')

def smart_savgol(lc,window_size):
  y=lc.get_y()
  ws=int(window_size)
  if ws%2==0:
    ws+=1
  with wr.catch_warnings():
    wr.filterwarnings('ignore')
    return sgnl.savgol_filter(y,ws,3)

def time_median_smooth(lc,window_size):
  y=lc.get_y()
  x=lc.get_x()
  ws2=window_size/2.0
  length=len(x)
  newy=np.zeros(length)
  for i in range(length):
    mask=np.logical_and(x>=x[i]-ws2,x<x[i]+ws2)
    newy[i]=np.median(y[mask])
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
  return (np.mean(x**2))**0.5
