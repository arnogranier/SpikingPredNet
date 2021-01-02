import brian2 as b2 
import numpy as np 

from neuron_model import C, gL, EL, VT, DeltaT, Vcut
from neuron_model import neurons


def Addconverge(net:b2.Network, a:str, b:str, centerw:float, surroundw:float):
    """Add 2D converging synapses from population a to b, with two weights:
    one for the central one-to-one synapses, and one for the surround 
    synapses. Think LGN->V1 or convolutions with 2 weights.

    Args:
        net (b2.Network): The brian2.Network containing the two populations, 
                          and to which we add the synapses
        a (str): Name of the projecting population
        b (str): Name of the receiving populations
        centerw (float): Weight of the center connections
        surroundw (float): Weight of the surround connections
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
    
def AddOnCenterOffSuroundRetina(net:b2.Network, a:str, w:tuple=(12,-4),
                                name:str='retina'):
    """Add a population receiving on-center off-surround local converging
    afference from population a. If a is pixel-space, then this population
    models the activity of retinal ganglion cells (or LGN cells)

    Args:
        net (b2.Network): The brian2.Network containing population a, 
                          and to which we add the new population and synapses
        a (str): Name of the projecting population
        w (tuple, optional): (weight of central connection,
                              weight of surround connections).
                             Defaults to (12,-4).
        name (str, optional): Name of the receiving population. Defaults to
                              'retina'.
    """
    
    S = net[a]
    retina = neurons(S.N, name=name)
    net.add(retina)
    Addconverge(net, a, name, w[0], w[1])
    