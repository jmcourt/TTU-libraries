import data_lib as dat
import file_lib as fi
from   matplotlib import pyplot as pl
import numpy as np
from   requests import exceptions as rx
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

# ========= BASE STACKED IMAGE OBJECT! =====================================================================================================================

class stacked_image(dat.DataSet):

  # Basic initialisation of all stacked images.  Gets times, images, x and y coord arrays and meta, and attempts to work out units

  @fi.mjit()
  def __init__(self,t,x,y,i,meta={}):
    ts=np.array(t).astype(float)
    ims=np.array(i).astype(int)
    #sort data by time
    sorted_args=ts.argsort()
    self.t=ts[sorted_args]
    self.i=ims[sorted_args]
    self.x=x
    self.y=y
    self.meta=meta
    self.unpack_metadata()

  # Simple plotter

  def quickplot(self,output=None,block=False,cmap='viridis',title=True,**kwargs):
    ax=fi.filter_axes(output)
    self.vertical_mean().lin_colour_plot(ax=ax,cmap=cmap,**kwargs)
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    if title:
      ax.set_title(self.meta['name'])
    fi.plot_save(output,block)

  # Simple data processes

  def vertical_sum(self):
    sum_i=np.sum(self.i,axis=0)
    return dat.TwoD_Dataframe(self.x,self.y,sum_i)

  def vertical_mean(self):
    sum_i=np.mean(self.i,axis=0)
    return dat.TwoD_Dataframe(self.x,self.y,sum_i)

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
  imghead=f[1].header
  q=imgdat['QUALITY']
  mask=q==0
  t=imgdat['TIME'][mask]
  i=imgdat['RAW_CNTS'][mask]

  # WARNING!  RA and DEC treated as orthonormal, cartesian.  This approximation will be bad near the poles

  x_scale=imghead['1CDLT4']
  y_scale=imghead['2CDLT4']
  x_0=imghead['1CRVL4']-(imghead['1CRPX4']*x_scale)
  y_0=imghead['2CRVL4']-(imghead['2CRPX4']*x_scale)
  xdim=int(imghead['TDIM7'].split(',')[0].split('(')[1])
  ydim=int(imghead['TDIM7'].split(',')[1].split(')')[0])
  x=np.arange(0,xdim,1,dtype=int)*x_scale+x_0
  y=np.arange(0,ydim,1,dtype=int)*y_scale+y_0

  f.close()
  return stacked_image(t,x,y,i,meta=imeta)

#  Wrap the astropy warning with the fits.open call to save adding this warning to eeeeevery loading routine.

def safe_open_fits(filename):
  if not imported_astropy:
    raise ImportError('No Astropy module found!  Cannot open fits files!')
  return fits.open(filename)
 
