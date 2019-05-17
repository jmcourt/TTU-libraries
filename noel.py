import numpy as np
from astroML.time_series import  lomb_scargle
from astroML.utils import check_random_state
   
def LS_bootstrap_err_est(t, y, dy, omega,
                           generalized=True, subtract_mean=True,
                           N_bootstraps=1000, random_state=None,
                           hist=False, plot_hist=True,Nbins=200):
    """Use a bootstrap analysis to compute Lomb-Scargle error estimation

    Parameters
    ----------
    The first set of parameters are passed to the lomb_scargle algorithm

    t : array_like
        sequence of times
    y : array_like
        sequence of observations
    dy : array_like
        sequence of observational errors
    omega : array_like
        frequencies at which to evaluate p(omega)
    generalized : bool
        if True (default) use generalized lomb-scargle method
        otherwise, use classic lomb-scargle.
    subtract_mean : bool
        if True (default) subtract the sample mean from the data before
        computing the periodogram.  Only referenced if generalized is False

    Remaining parameters control the bootstrap

    N_bootstraps : int
        number of bootstraps
    random_state : None, int, or RandomState object
        random seed, or random number generator

    Returns
    -------
    D : ndarray
        distribution of the height of the highest peak
    """
    random_state = check_random_state(random_state)
    t = np.asarray(t)
    y = np.asarray(y)
    dy = np.asarray(dy) + np.zeros_like(y)

    D = np.zeros(N_bootstraps)
    omegaD= np.zeros(N_bootstraps)
   
    for i in range(N_bootstraps):
        ind = random_state.randint(0, len(y), len(y))
        p = lomb_scargle(t[ind], y[ind], dy[ind], omega,
                         generalized=generalized, subtract_mean=subtract_mean)
        D[i] = p.max()
        omegaD[i]=omega[p.argmax()]
           
           
    if hist:
       
        if plot_hist:
            from matplotlib import pyplot as plt
            frecD=omegaD.copy()/(2*np.pi)
           
            plt.figure('bootstrap hist')
            plt.hist(frecD,normed=True, bins=Nbins)
            plt.hist(frecD,normed=True,histtype='step')

            plt.figure('bootstrap cumhist')
            Xcum=np.sort(D)
            Ycum=np.array(range(N_bootstraps))/float(N_bootstraps)
            plt.plot(Xcum,Ycum)
            #plt.xlim(Xcum,Xcum)
            plt.grid()
            plt.minorticks_on()
            plt.xlabel('')

        return D,omegaD
    else:
        return D 
