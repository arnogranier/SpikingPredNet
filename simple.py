import brian2 as b2 
import brian2genn 
import numpy as np 
import matplotlib.pyplot as plt 

from neuron_model import C, gL, EL, VT, DeltaT, Vcut
from neuron_model import taupre, taupost, Apre, Apost
from neuron_model import neurons, synapses
from plots import rplots
from modules import Area, connect 


def exp1():
    """
    
    """
    #
    b2.set_device('genn')
    
    #
    activationLOW = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0]])
    activationHIGH = np.array([[1, 0], [0, 1], [1, 0], [1, 0]])
    timedRatesLOW = b2.TimedArray(65*activationLOW*b2.Hz, dt=1*b2.second)
    timedRatesHIGH = b2.TimedArray(65*activationHIGH*b2.Hz, dt=1*b2.second)
    
    #
    net = b2.Network()
    
    #
    wINHIPE, wEXCIPE = -25, 12
    
    #
    W = np.array([[1, 0, 1], 
                  [0, 1, 1]])
    
    #
    LOW = Area(3, 'LOW', net, wINHIPE, wEXCIPE, IRPoisson=True,
               recordspikes=True)
    HIGH = Area(2, 'HIGH', net, wINHIPE, wEXCIPE, IRPoisson=True,
                recordspikes=True)
    
    #
    LOW.set_rates('timedRatesLOW')
    HIGH.set_rates('timedRatesHIGH') 
    # putting the var names as arguments of set_rates can seem a little off
    # putting and not really scalable to larger projects, but it's simply a 
    # workaround because b2genn does not support b2.TimedArray as model 
    # variables
    
    #
    connect(HIGH, LOW, W, wEXCIPE, plastic=False)
    
    #
    net.run(4*b2.second)
    
    #
    rplots(LOW['IR'], HIGH['IR'], LOW['PPE'], LOW['NPE'])
    plt.show()

def exp2():
    """
    
    """
    
    b2.set_device('genn')
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
    """
    
    """
    
    b2.set_device('genn')
    activationHIGH = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    timedRatesHIGH = b2.TimedArray(65*activationHIGH*b2.Hz, dt=1*b2.second)
    net = b2.Network()
    wINHIPE, wEXCIPE = -20, 12
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
    """
    
    """
    
    indices = np.tile(np.array([0, 2, 1, 3, 2, 4, 3, 5]), 25)
    times = np.tile(np.array([1, 1, 21, 21, 41, 41, 61, 61]), 25)
    times += 80*np.repeat(np.arange(0,25), 8)
    times[-40:] += 200
    times = times * b2.ms
    net = b2.Network()
    wINHIPE, wEXCIPE = -30, 35
    wINHIIR, wEXCIIR = -20, 20
    W = np.array([[1, 0, 1, 0, 0, 0], 
                  [0, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 1]])
    LOW = Area(6, 'LOW', net, wINHIPE, wEXCIPE, IRSet=True, recordspikes=True)
    HIGH = Area(4, 'HIGH', net, wINHIPE, wEXCIPE, wINHIIR, wEXCIIR,
                onlyIR=True, lateralplasticity=True, recordspikes=True, 
                SetSpikes = [(0,1.8)])
    LOW.set_spikes(indices, times)
    connect(HIGH, LOW, W, wEXCIPE, wEXCIIR, plastic=False, onlyNPE=True)
    net['s_HIGH_IR_LOW_interNPE'].w_syn = 0*b2.mV
    net['s_HIGH_IR_HIGH_IR_P'].thetaSTDP = 40*b2.mV
    net.run(1.6*b2.second)
    net['s_HIGH_IR_LOW_interNPE'].w_syn = 35*b2.mV
    net['s_HIGH_IR_HIGH_IR_P'].thetaSTDP = 7*b2.mV
    net.run(.6*b2.second)
    rplots(LOW['IR'], HIGH['IR'], LOW['NPE'])
    plt.show()

def exp5():
    """
    
    """
    
    pass 

if __name__ == "__main__":
    exp5()