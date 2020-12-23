import brian2 as b2 
from neuron_model import C, gL, EL, VT, DeltaT, Vcut
from neuron_model import neurons
import numpy as np 

def Addconverge(net, a, b, centerw, surroundw, overlap=True):
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
    i = np.concatenate([s.flatten() for s in [ku, kd, kr, kl, kur, kul, kdr, kdl]])
    j = np.concatenate([s.flatten() for s in [u, d, r, l, ur, ul, dr, dl]])
    surround.connect(i=i, j=j)
    net.add(surround)
    
def AddOnCenterOffSuroundRetina(net, s, wh=None, name=''):
    S = net[s]
    retina = neurons(S.N, name='retina')
    net.add(retina)
    Addconverge(net, s, 'retina', 12, -4)
    