import numpy as np

# ======= Fourier normalisation procedures =======

'''Input key:
    counts : the total number of counts in the lightcurve from which the FT was taken
    rate   : the source + background count rate (per unit time) of the data sample from which the
              spectrum was created.
    bg_rate: the background count rate (per unit time) of the data sample from which the spectrum
              was created.
    const  : the average power given in Leahy normalisation for pure white noise.  Theoretically
              const=2, but in practice is slightly lower and varies between telescopes.'''

def rms(f,counts,rate,bg_rate,const=2):
  return leahy2rms(leahy(f,counts),rate,bg_rate,const)

def leahy(f,counts):
  return 2.0*f/counts

def leahy2rms(l,rate,bg_rate,const=2):
   denom=(rate-bg_rate)**2
   if denom<=0:
      raise DataError('RMS normalisation received source+bg rate smaller than bg rate!')
   else:
      mult=1.0/denom
   rms=(l-const)*(bg_rate)*mult
   return rms

# ======== Generalised Lomb-Scargling function =========

def lomb_scargle(t,y,ye,freqs,norm='auto'):

    # Generalised L-S from Zechmeister & Kuerster, 2009, eq 5-15

    w=(ye**-2)/(np.sum(ye**-2))
    om=freqs*2*np.pi

    ft_table=np.array(np.outer(om,t))

    # ^ Vectorising the operation.  Axis=1 means we are summing over t, not over om

    Y=np.sum(w*y)
    C=np.sum(w*np.cos(ft_table),axis=1)
    S=np.sum(w*np.sin(ft_table),axis=1)

    YhY=np.sum(w*y**2)
    YhC=np.sum(w*y*np.cos(ft_table),axis=1)
    YhS=np.sum(w*y*np.sin(ft_table),axis=1)
    ChC=np.sum(w*np.cos(ft_table)**2,axis=1)
    ChS=np.sum(w*np.cos(ft_table)*np.sin(ft_table),axis=1)
    ShS=np.sum(w*np.sin(ft_table)**2,axis=1)

    YY=YhY-Y*Y
    YC=YhC-Y*C
    YS=YhS-Y*S
    SS=ShS-S*S
    CS=ChS-C*S
    CC=ChC-C*C

    D=CC*SS-CS**2

    #        V  normalisation from Z&K09 and from Horne & Baliunas 86

    if norm=='auto':
      norm=(len(t)-1)/2

    return norm*1/(YY*D)*(SS*(YC**2)+CC*(YS**2)-2*CS*YC*YS)
