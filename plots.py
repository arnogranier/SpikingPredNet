import brian2 as b2
import matplotlib.pyplot as plt 
import numpy as np 

def raster_plot(M:b2.SpikeMonitor):
    """[summary]

    Args:
        M (b2.SpikeMonitor): [description]

    Returns:
        [type]: [description]
    """
    return b2.plot(M.t/b2.ms, M.i, ',')

def rateplot2d(M:b2.SpikeMonitor, from_:float, to:float, wh:tuple):
    """[summary]

    Args:
        M (b2.SpikeMonitor): [description]
        from_ (float): [description]
        to (float): [description]
        wh (tuple): [description]

    Returns:
        [type]: [description]
    """
    im1 = M.i[np.where(np.logical_and(from_<M.t/b2.ms, M.t/b2.ms<to))[0]]
    m = np.max(im1)
    idx = np.concatenate([np.bincount(im1), np.zeros(wh[0]*wh[1]-m-1)])
    idx = idx.reshape(wh[0],wh[1])
    return plt.imshow(idx)     

def rplots(*args):
    """[summary]
    
    Args:
        Any number of b2.SpikeMonitor
    """
    for arg in args:
        plt.figure()
        raster_plot(arg)

def vplots(vname:str, *args):
    """[summary]
    
    Args:
        vname (str): [description]
        Any number of b2.StateMonitor
    """
    for arg in args:
        plt.figure()
        N = len(arg.vm)
        for i in range(N):
            plt.plot(arg.t/b2.ms, arg.getattr(vname)[i]/b2.mV)
            