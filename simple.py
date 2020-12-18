import brian2 as b2 
import brian2genn 
b2.set_device('genn')
import numpy as np 

from neuron_model import neurons, C, gL, taum, EL, VT, DeltaT, Vcut
from plots import raster_plot

hIR = neurons(2, name='hIR')
interhIR = neurons(2, name='interhIR')
lPPE = neurons(5, name='lPPE')
interlPPE = neurons(5, name='interlPPE')
lNPE = neurons(5, name='lNPE')
interlNPE = neurons(5, name='interlNPE')
rates = b2.TimedArray(np.tile(np.array([[80, 0], [0, 80]]), (10,1))*b2.Hz, dt=1000.*b2.ms)
lIR = b2.PoissonGroup(5, rates='rates(t)', name='lIR')

<<<<<<< HEAD
b2.Synapses(hIR, lPPE)
b2.Synapses(hIR, interlNPE)
b2.Synapses(lPPE, interhIR)
b2.Synapses(lNPE, hIR)
b2.Synapses(lIR, lNPE)
b2.Synapses(lIR, interlPPE)
=======
>>>>>>> 2ec040a9b685d425cb582d23c36c561ea95b51ee
