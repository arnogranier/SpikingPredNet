import brian2 as b2 
import brian2genn 
b2.set_device('genn')
import numpy as np 
import matplotlib.pyplot as plt 

from neuron_model import C, gL, EL, VT, DeltaT, Vcut
from neuron_model import neurons, synapses
from plots import rplots, vplots
from area import Area, connect 

def exp1():
    activationLOW = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 1], [0, 1, 0]])
    activationHIGH = np.array([[0, 1], [1, 0], [0, 1], [0, 1]])
    timedRatesLOW = b2.TimedArray(65*activationLOW*b2.Hz, dt=1*b2.second)
    timedRatesHIGH = b2.TimedArray(65*activationHIGH*b2.Hz, dt=1*b2.second)
    net = b2.Network()
    wINHIPE, wEXCIPE = -20, 12
    W = np.array([[1, 0, 1], 
                  [0, 1, 1]])
    LOW = Area(3, 'LOW', net, wINHIPE, wEXCIPE, IRPoisson=True, recordspikes=True)
    HIGH = Area(2, 'HIGH', net, wINHIPE, wEXCIPE, IRPoisson=True, recordspikes=True)
    LOW.set_rates('timedRatesLOW')
    HIGH.set_rates('timedRatesHIGH') # putting the var names here can seem a little off putting
                                     # and not really scalable to larger projects, but it's
                                     # simply a workaround because b2genn does not support
                                     # b2.TimedArray as model variables
    connect(HIGH, LOW, W, wEXCIPE)
    net.run(4*b2.second)
    rplots(LOW['IR'], HIGH['IR'], LOW['PPE'], LOW['NPE'])
    plt.show()


if __name__ == "__main__":
    exp1()