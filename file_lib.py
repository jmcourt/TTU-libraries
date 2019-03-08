from matplotlib import pyplot as pl

def plot_save(filename):
  if filename==None:
    pl.show(block=False)
  else:
    pl.savefig(filename)
    pl.close()
