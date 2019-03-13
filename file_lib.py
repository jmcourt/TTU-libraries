#Routines for file management and plot management

from matplotlib import pyplot as pl
from matplotlib import axes as xs

def plot_save(output):
  if output==None:
    pl.show(block=False)
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
