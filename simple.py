import brian2 as b2 
import brian2genn 
b2.set_device('genn')
import numpy as np 
import matplotlib.pyplot as plt 

from neuron_model import C, gL, EL, VT, DeltaT, Vcut
from neuron_model import neurons, synapses
from plots import rplots, vplots
from modules import Area, connect 

def exp1():
    activationLOW = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0]])
    activationHIGH = np.array([[1, 0], [0, 1], [1, 0], [1, 0]])
    timedRatesLOW = b2.TimedArray(65*activationLOW*b2.Hz, dt=1*b2.second)
    timedRatesHIGH = b2.TimedArray(65*activationHIGH*b2.Hz, dt=1*b2.second)
    net = b2.Network()
    wINHIPE, wEXCIPE = -25, 12
    W = np.array([[1, 0, 1], 
                  [0, 1, 1]])
    LOW = Area(3, 'LOW', net, wINHIPE, wEXCIPE, IRPoisson=True, recordspikes=True)
    HIGH = Area(2, 'HIGH', net, wINHIPE, wEXCIPE, IRPoisson=True, recordspikes=True)
    LOW.set_rates('timedRatesLOW')
    HIGH.set_rates('timedRatesHIGH') # putting the var names here can seem a little off putting
                                     # and not really scalable to larger projects, but it's
                                     # simply a workaround because b2genn does not support
                                     # b2.TimedArray as model variables
    connect(HIGH, LOW, W, wEXCIPE, plastic=False)
    net.run(4*b2.second)
    rplots(LOW['IR'], HIGH['IR'], LOW['PPE'], LOW['NPE'])
    plt.show()

def exp2():
    activationLOW = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1]])
    timedRatesLOW = b2.TimedArray(65*activationLOW*b2.Hz, dt=1*b2.second)
    net = b2.Network()
    wINHIPE, wEXCIPE = -25, 12
    wINHIIR, wEXCIIR = -20, 25
    W = np.array([[1, 0, 1], 
                  [0, 1, 1]])
    LOW = Area(3, 'LOW', net, wINHIPE, wEXCIPE, IRPoisson=True, recordspikes=True)
    HIGH = Area(2, 'HIGH', net, wINHIPE, wEXCIPE, wINHIIR, wEXCIIR, onlyIR=True, recordspikes=True)
    LOW.set_rates('timedRatesLOW')
    connect(HIGH, LOW, W, wEXCIPE, wEXCIIR, plastic=False)
    net.run(4*b2.second)
    rplots(LOW['IR'], HIGH['IR'], LOW['PPE'], LOW['NPE'])
    plt.show()

def exp3():
    activationHIGH = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    timedRatesHIGH = b2.TimedArray(65*activationHIGH*b2.Hz, dt=1*b2.second)
    net = b2.Network()
    wINHIPE, wEXCIPE = -25, 12
    wINHIIR, wEXCIIR = -20, 25
    W = np.array([[1, 0, 1], 
                  [0, 1, 1]])
    LOW = Area(3, 'LOW', net, wINHIPE, wEXCIPE, wINHIIR, wEXCIIR, recordspikes=True)
    HIGH = Area(2, 'HIGH', net, wINHIPE, wEXCIPE, IRPoisson=True, onlyIR=True, recordspikes=True)
    HIGH.set_rates('timedRatesHIGH')
    connect(HIGH, LOW, W, wEXCIPE, wEXCIIR, plastic=False)
    net.run(4*b2.second)
    rplots(LOW['IR'], HIGH['IR'], LOW['PPE'], LOW['NPE'])
    plt.show()

def exp4():
    pass 

def exp5():
    pass 

if __name__ == "__main__":
    exp2()