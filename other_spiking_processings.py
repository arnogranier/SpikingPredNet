import brian2 as b2 
import numpy as np 

from neuron_model import C, gL, EL, VT, DeltaT, Vcut
from neuron_model import neurons


def Addconverge(net:b2.Network, a:str, b:str, centerw:float, surroundw:float,
                overlap:bool=True):
    """[summary]

    Args:
        net (b2.Network): [description]
        a (str): [description]
        b (str): [description]
        centerw (float): [description]
        surroundw (float): [description]
        overlap (bool, optional): [description]. Defaults to True.
    """
    
    A = net[a]
    B = net[b]
    center = b2.Synapses(A, B, on_pre='vm+=%s*mV'%centerw)
    surround = b2.Synapses(A, B, on_pre='vm+=%s*mV'%surroundw)
    center.connect('i==j')
    net.add(center)
    N = A.N
    sN = int(np.sqrt(N))
    k = np.arange(N).reshape(sN, sN)
    u, ku = k[1:,:]-sN, k[1:,:]
    d, kd = k[:sN-1,:]+sN, k[:sN-1,:] 
    r, kr = k[:,:sN-1]+1, k[:,:sN-1]
    l, kl = k[:,1:]-1, k[:,1:]
    ur, kur = k[1:, :sN-1]-sN+1, k[1:, :sN-1]
    ul, kul = k[1:, 1:]-sN-1, k[1:, 1:]
    dr, kdr = k[:sN-1, :sN-1]+sN+1, k[:sN-1, :sN-1]
    dl, kdl = k[:sN-1, 1:]+sN-1, k[:sN-1, 1:]
    i = np.concatenate([s.flatten() for s in [ku, kd, kr, kl, kur,
                                              kul, kdr, kdl]])
    j = np.concatenate([s.flatten() for s in [u, d, r, l, ur, ul, dr, dl]])
    surround.connect(i=i, j=j)
    net.add(surround)
    
def AddOnCenterOffSuroundRetina(net:b2.Network, s:str, w:tuple=(12,-4),
                                name:str='retina'):
    """[summary]

    Args:
        net (b2.Network): [description]
        s (str): [description]
        wh (tuple, optional): [description]. Defaults to (12,-4).
        name (str, optional): [description]. Defaults to 'retina'.
    """
    
    S = net[s]
    retina = neurons(S.N, name=name)
    net.add(retina)
    Addconverge(net, s, 'retina', wh[0], wh[1])
    