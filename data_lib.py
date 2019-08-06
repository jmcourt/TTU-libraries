import copy
from matplotlib import colors as co
from matplotlib import pyplot as pl
import numpy as np
from numpy import random as rn
import warnings as wr

class DataError(Exception):
  pass

# a base class with a bunch of procedures common to lc and img datasets

class DataSet(object):
  def unpack_metadata(self):
    if 'name' in self.meta.keys():
      self.objname=self.meta['name']
    else:
      self.objname='unknown'
    if 'mission' in self.meta.keys():
      self.mission=self.meta['mission']
    else:
      self.mission='unknown'

  def copy(self):
    return copy.deepcopy(self)

  def is_empty(self):
    return self.get_length()==0

class TwoD_Dataframe(object):
  def __init__(self,xvalues,yvalues,zvalues,meta={}):
    xlen=len(xvalues)
    ylen=len(yvalues)
    zdims=np.shape(zvalues)
    if len(zdims)!=2:
      raise DataError(str(z)+'D data cannot be put into a TwoD_Dataframe!')
    zxlen=zdims[1]
    zylen=zdims[0]

    if xlen==zxlen and ylen==zylen:
      self.z=zvalues
    elif xlen==zylen and ylen==zxlen:
      wr.warn('Matrix appears to be the wrong way round!  Taking transverse for TwoD_Dataframe!')
      self.z=zvalues.T
    else:
      raise DataError('Matrix does not match x and y dimensions in TwoD_Dataframe!')
    self.x=xvalues
    self.y=yvalues
    if 'x_dimension' in meta.keys():
      self.x_dim=meta['x_dimension']
    else:
      self.x_dim=''
    if 'y_dimension' in meta.keys():
      self.y_dim=meta['y_dimension']
    else:
      self.y_dim=''
    if 'x_unit' in meta.keys():
      self.x_unit=meta['x_unit']
    else:
      self.x_unit=''
    if 'y_unit' in meta.keys():
      self.y_unit=meta['x_unit']
    else:
      self.y_unit=''

  def copy(self):
    return copy.deepcopy(self)

  def xdir_norm(self):
    for i in range(len(self.z[:,0])):
      row=self.z[i,:]
      mask=np.logical_not(np.isnan(row))
      if np.sum(mask)==0:
        continue
      mean=np.mean(row[mask])
      std=np.std(row[mask])
      self.z[i,:]=(self.z[i,:]-mean)/std

  def ydir_norm(self):
    for i in range(len(self.z[0,:])):
      col=self.z[:,i]
      mask=np.logical_not(np.isnan(col))
      if np.sum(mask)==0:
        continue
      mean=np.mean(col[mask])
      std=np.std(col[mask])
      self.z[:,i]=(self.z[:,i]-mean)/std

  def get_x(self):
    return self.x
  def get_y(self):
    return self.y
  def get_xy_stack(self):
    X,Y=np.meshgrid(self.get_x(),self.get_y())
    return np.vstack((X.ravel(),Y.ravel()))
  def get_data(self):
    return self.z
  def get_z(self):
    return self.z
  def get_nanless_z(self):
    mask=np.logical_not(np.isnan(self.z))
    return self.z[mask]

  def get_mean(self):
    return np.mean(self.get_nanless_z())
  def get_median(self):
    return np.median(self.get_nanless_z())
  def get_std(self):
    return np.std(self.get_nanless_z())
  def get_max(self):
    return np.max(self.get_nanless_z())
  def get_min(self):
    return np.min(self.get_nanless_z())
  def get_nozero_min(self):
    return np.min(self.get_nanless_z()[self.get_nanless_z()>0])

  def log_colour_plot(self,colour_min=None,colour_max=None,ax=None,cmap='viridis'):
    if colour_min==None:
      colour_min=self.get_nozero_min()
    if colour_max==None:
      colour_max=self.get_max()
    if ax==None:
      ax=pl.gca()
    colourplot=ax.pcolor(self.x,self.y,self.z,cmap=cmap,norm=co.LogNorm(vmin=colour_min,vmax=colour_max))
    return colourplot

  def lin_colour_plot(self,colour_min=None,colour_max=None,ax=None,cmap='viridis',**kwargs):
    if colour_min==None:
      colour_min=self.get_min()
    if colour_max==None:
      colour_max=self.get_max()
    if ax==None:
      ax=pl.gca()
    colourplot=ax.pcolor(self.x,self.y,self.z,cmap=cmap,norm=co.Normalize(vmin=colour_min,vmax=colour_max),**kwargs)
    return colourplot

  def dump(self,filename,header=True):
    f=open(filename,'w')
    if header:
      if self.x_dim=='':
        xlabel='X'
      else:
        xlabel=self.x_dim.upper()
      if self.y_dim=='':
        ylabel='Y'
      else:
        ylabel=self.y_dim.upper()
      f.write(xlabel+'_AXIS\n')
      for i in self.x:
        f.write(str(i)+'\n')
      f.write(ylabel+'_AXIS\n')
      for i in self.y:
        f.write(str(i)+'\n')
      f.write('DATA_('+ylabel+'_INCREASES_DOWN,_'+xlabel+'_INCREASES_TO_RIGHT)\n')
    for row in self.z:
      write_string=''
      for val in row:
        write_string+=str(val)+','
      f.write(write_string[:-1]+'\n')
    f.close()

def rebin(binsize,x,y,ye=None):

  # Shameless ripoff from pan_lib

  binlx=binsize*np.floor(x[0]/binsize)                                   # Initialising 'bin lowest x', or the lowest x value of the current bin
  binct=0.0                                                              # Initialising 'bin count', or number of values sorted into the current bin

  xb=[x[0]]                                                              # Setting up arrays to append binned values into
  yb=[0]   
  yeb=[0]

  for xid in range(len(x)):

    if x[xid]-binlx < binsize:                                          ## If the difference between the current x and bin start x is less than bin width:

      binct+=1
      yb[-1]=yb[-1]+y[xid]                                                  #  Add y to current bin
      yeb[-1]=yeb[-1]+(ye[xid]**2)                                          #  Add y error in quadrature to current bin
 
    else:                                                               ## Otherwise:
      while binlx+binsize<x[xid]:
        binlx+=binsize                                                      #  Create new bin with minimum x equal to current x
      xb.append(binlx)                                                      #  Append next x value into new array element
      yb[-1]=yb[-1]/binct                                                   #  Divide y in previous bin by bincount to get the average
      yeb[-1]=(np.sqrt(yeb[-1]))/binct                                      #  Sqrt error and divide by bincount
      yb.append(y[xid])                                                     #  Append current y value into new array element
      yeb.append((ye[xid])**2)

      binct=1                                                               #  Reset bin count to 1

  yb[-1]=yb[-1]/binct                                                    ## Clean up final bin
  yeb[-1]=(np.sqrt(yeb[-1]))/binct
  return np.array(xb),np.array(yb),np.array(yeb)

def rebin_by_factor(bin_factor,x,y,ye=None):
  x=np.array(x)
  y=np.array(y)
  if type(ye)==type(None):
    ye=[0]*len(x)
  ye=np.array(ye)
  bin_factor=int(bin_factor)
  if bin_factor>len(x):
    raise DataError('Bin factor greater than number of datapoints!')
  x_overshoot=len(x)%bin_factor
  masked_xs=[0]*bin_factor
  masked_ys=[0]*bin_factor
  masked_yes=[0]*bin_factor
  if x_overshoot>0:   # cropping out data which would fall outside of any of the new bins
    x=x[:-x_overshoot]
    y=y[:-x_overshoot]
    ye=ye[:-x_overshoot]
  index_range=np.array(range(len(x)))
  for i in range(bin_factor):
    mask=index_range%bin_factor==i
    masked_xs[i]=x[mask]
    masked_ys[i]=y[mask]
    masked_yes[i]=ye[mask]
  xmatrix=np.vstack(masked_xs)
  ymatrix=np.vstack(masked_ys)
  yematrix=np.vstack(masked_yes)
  binned_x=np.mean(xmatrix,axis=0)
  binned_y=np.mean(ymatrix,axis=0)
  binned_ye=np.sqrt(np.sum(yematrix**2,axis=0))/bin_factor

  return binned_x,binned_y,binned_ye

def log_rebin(log_res,x,y,ye=None):
  # shamelessly repurposed from pan_lib
  x=np.array(x)
  y=np.array(y)
  if type(ye)==type(None):
    ye=[0]*len(x)
  ye=np.array(ye)
  lin_res=x[1]-x[0]
  hinge=(lin_res*10**log_res)/((10**log_res)-1)                      # Find the 'hinge' point at which to switch between linear and logarithmic binning
  lbin=np.log10(x[0])
  xb =10**(np.arange(lbin,np.log10(x[-1]),log_res))                       # Setting up arrays to append binned values into
  yb =np.zeros(len(xb))
  yeb=np.zeros(len(xb))

  hingel=sum((xb)<=hinge)                                                # Getting the ID of the hinge-point in the log

  xbl=len(xb)

  for i in range(hingel,xbl):

    lowid=int(((10**((i*log_res)+lbin))-x[0])/lin_res)               # Calculate the ID of the lowest linear bin that corresponds to this log bin
    uppid=int(((10**(((i+1)*log_res)+lbin))-x[0])/lin_res)           # Calculate the ID of the highest linear bin that corresponds to this log bin
    if uppid>lowid:
      yb[i]=np.mean(y[lowid:uppid])
      yeb[i]=(np.sqrt(sum(np.array(ye[lowid:uppid])**2)))/int(uppid-lowid)
    else:
      yb[i]=0                                                          # If no data found, error=power=0
      yeb[i]=0

  mask=x<hinge
  lmask=xb>hinge

  binned_x=np.append(x[mask],xb[lmask])
  binned_y=np.append(y[mask],yb[lmask])
  binned_ye=np.append(ye[mask],yeb[lmask])

  return binned_x,binned_y,binned_ye

# --------------- Bootstrapping methods ------------------

def gaussian_bootstrap(y,ye):
  assert len(y)==len(ye)
  return rn.normal(loc=y,scale=ye)
