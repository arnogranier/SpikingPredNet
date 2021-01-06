import brian2 as b2
import matplotlib.pyplot as plt 
import numpy as np 

def raster_plot(M:b2.SpikeMonitor, marker:str=','):
    """Raster plot of a brian2.SpikeMonitor

    Args:
        M (b2.SpikeMonitor): Recording of spikes
        marker (str): The marker to use in the plot
    Returns:
        list of plt.Line2D: The raster plot
    """
    return b2.plot(M.t/b2.ms, M.i, marker)

def rateplot2d(M:b2.SpikeMonitor, from_:float, to:float, wh:tuple):
    """2D plot of firing rates. This is particularly useful to visualize 
    activity in a spiking population processing 2D data such as images.

    Args:
        M (b2.SpikeMonitor): Recording of spikes
        from_ (float): Time when we start taking spikes into account
        to (float): Time when we stop taking spikes into account
        wh (tuple): (width, height) of the data. w*h must be nb of neurons

    Returns:
        plt.AxesImage: The 2D rate plot
    """
    im1 = M.i[np.where(np.logical_and(from_<M.t/b2.ms, M.t/b2.ms<to))[0]]
    m = np.max(im1)
    idx = np.concatenate([np.bincount(im1), np.zeros(wh[0]*wh[1]-m-1)])
    idx = idx.reshape(wh[0],wh[1])
    return plt.imshow(idx)     

def rplots(*args, ma):
    """Multiple raster plots. Calls raster_plot for each argument
    
    Args:
        Any number of b2.SpikeMonitor
    """
    for arg in args:
        plt.figure()
        raster_plot(arg)

def vplots(vname:str, *args):
    """Multiple plots of the evolution of a variable in different
    brian2.StateMonitor.
    
    Args:
        vname (str): Name of the variable
        Any number of b2.StateMonitor
    """
    for arg in args:
        plt.figure()
        N = len(arg.vm)
        for i in range(N):
            plt.plot(arg.t/b2.ms, arg.getattr(vname)[i]/b2.mV)
            