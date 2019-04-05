import copy
from matplotlib import colors as co
from matplotlib import pyplot as pl
import numpy as np
import warnings as wr

class DataError(Exception):
  pass

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

  def lin_colour_plot(self,colour_min=None,colour_max=None,ax=None,cmap='viridis'):
    if colour_min==None:
      colour_min=self.get_min()
    if colour_max==None:
      colour_max=self.get_max()
    if ax==None:
      ax=pl.gca()
    colourplot=ax.pcolor(self.x,self.y,self.z,cmap=cmap,norm=co.Normalize(vmin=colour_min,vmax=colour_max))
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
