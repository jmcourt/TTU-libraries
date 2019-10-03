import lc_lib as lcl

def load_source(name,normed=False):
  lastsector=15
  obdirs={}
  flist=open('/home/jamie/data/TESS/_index.txt')
  for line in flist:
    l=line.split(':')
    obdirs[tuple(l[0].split(';'))]=l[1][:-1]

  init=False
  for i in range(1,lastsector+1):
    try:
      filnam=obdirs[(name,str(i))]
    except KeyError:
      try:
        filnam=obdirs[('V* '+name,str(i))]
      except KeyError:
        try:
          filnam=obdirs[('SV* '+name,str(i))]
        except KeyError:
          try:
            filnam=obdirs[('[CCH78] '+name,str(i))]
          except KeyError:
            continue
    t_lc=lcl.get_tess_lc('/home/jamie/data/TESS/'+filnam)
    if normed:
      t_lc.add_constant(-t_lc.get_mean())
      t_lc.multiply_by_constant(1/t_lc.get_std())
    if not init:
      init=True
      lc=t_lc
    else:
      lc.add_data(t_lc)
  if init:
    return lc
  else:
    raise IOError
