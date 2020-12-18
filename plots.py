import brian2 as b2
import matplotlib.pyplot as plt 
import numpy as np 

def raster_plot(M):
    return b2.plot(M.t/b2.ms, M.i, ',')

def rateplot2d(M, from_, to, wh):
    im1 = M.i[np.where(np.logical_and(from_<M.t/b2.ms, M.t/b2.ms<to))[0]]
    m = np.max(im1)
    idx = np.concatenate([np.bincount(im1), np.zeros(wh[0]*wh[1]-m-1)])
    idx = idx.reshape(wh[0],wh[1])
    return plt.imshow(idx)     