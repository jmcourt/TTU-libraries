#Routines for file management, plot management and some technical backends

from matplotlib import pyplot as pl
from matplotlib import axes as xs

# a conditional jit compiler to avoid having numba be mandatory

try:
  import numba as nb
  withjit=True
except ModuleNotFoundError:
  withjit=False

class mjit(object):
  def __call__(self, f):
    if not withjit:
      return f
    else:
      return nb.jit(f)

def is_iterable(obj):
  try:
    iter(obj)
    return True
  except:
    return False

def plot_save(output,block=False):
  if output==None:
    pl.show(block=block)
  elif isinstance(output,xs._subplots.Axes):
    pass
  else:
    pl.savefig(str(output))
    pl.close()

def filter_axes(output):

  # specific function for some methods which can output to screen, a file or an axes of the user's choice.  Filters for axes objects: if one is found, it is returned.  If one is not found, a new figure is initiated and gca() axes are returned.

  if isinstance(output,xs._subplots.Axes):
    return output
  else:
    pl.figure()
    return pl.gca()
