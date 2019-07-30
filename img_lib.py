import data_lib as dat
import file_lib as fi
import fold_lib as fo
import freq_lib as frq
import func_lib as func
from   matplotlib import gridspec as gs
from   matplotlib import pyplot as pl
import numpy as np
import random as rn
from   requests import exceptions as rx
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

# ========= BASE STACKED IMAGE OBJECT! =====================================================================================================================

class stacked_image(dat.DataSet):

  # Basic initialisation of all lightcurves.  Gets x,y,ye and meta, and attempts to work out units

  @fi.mjit()
  def __init__(self,t,x,y,i,meta={}):
    ts=np.array(t).astype(float)
    ims=np.array(i).astype(int)
    #sort data by time
    sorted_args=tx.argsort()
    self.x=ts[sorted_args]
    self.i=ims[sorted_args]
    self.meta=meta
    self.unpack_metadata()

  # Generic metadata unpacking.  Use this in inherting classes to do fancy stuff with extra data



  # Simple plotter

  def quickplot(self,output=None,block=False,title=True,**kwargs):
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

  # Simple data processes


# TESS Target Pixel File opener

def get_tess_tpf(filename):
  if not imported_astropy:
    raise ImportError('No Astropy module found!  Cannot open fits files!')
  imeta={}
  f=safe_open_fits(filename)
  if f[1].header['TELESCOP'][:4].upper()!='TESS':
    raise dat.DataError('FITS file does not appear to be from TESS')
  if f[1].header['EXTNAME'].upper()!='PIXELS':
    raise dat.DataError('TESS FITS file does not appear to be a Target Pixel File')

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
  imgdat=f[1].data
  q=imgdat['QUALITY']
  mask=q==0
  x=imgdat['TIME'][mask]
  i=imgdat['RAW_CNTS'][mask]

  f.close()
  return stacked_image(x,i,meta=imeta)

#  Wrap the astropy warning with the fits.open call to save adding this warning to eeeeevery loading routine.

def safe_open_fits(filename):
  if not imported_astropy:
    raise ImportError('No Astropy module found!  Cannot open fits files!')
  return fits.open(filename)
 
