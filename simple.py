import brian2 as b2 
import brian2genn 
b2.set_device('genn')
import numpy as np 

from neuron_model import neurons, C, gL, taum, EL, VT, DeltaT, Vcut
from plots import raster_plot

hIR = neurons(2, name='hIR')
interhIRlocal = neurons(2, name='interhIRlocal')
interhIRglobal = neurons(1, name='interhIRglobal')
lPPE = neurons(5, name='lPPE')
interlPPE = neurons(5, name='interlPPE')
lNPE = neurons(5, name='lNPE')
interlNPE = neurons(5, name='interlNPE')
SetRates = b2.TimedArray(np.tile(np.array([[80, 0, 80, 0, 80], [0, 80, 80, 80, 0]]), (10,1))*b2.Hz, dt=1000.*b2.ms)
lIR = b2.NeuronGroup(5, 'rates : Hz', threshold='rand()<rates*dt', name='lIR')
lIR.run_regularly('rates = SetRates(t,i)', 1000*b2.ms)

W = np.array([[1, 0, 1, 0, 1],
              [0, 1, 1, 1, 0]])
sources, targets = W.nonzero()
shIRhIR = b2.Synapses(hIR, hIR, on_pre='vm+=11*mV')
shIRhIR.delay.delay='1ms+rand()'
shIRhIR.connect('i==j')
b2.Synapses(hIR, interhIRglobal, on_pre='vm+=11*mV').connect()
b2.Synapses(hIR, lPPE, on_pre='vm+=11*mV').connect(i=sources, j=targets)
b2.Synapses(hIR, interlNPE, on_pre='vm+=11*mV').connect(i=sources, j=targets)
b2.Synapses(lPPE, interhIRlocal, on_pre='vm+=11*mV').connect(i=targets, j=sources)
b2.Synapses(lNPE, hIR, on_pre='vm+=11*mV').connect(i=targets, j=sources)
b2.Synapses(lIR, lNPE, on_pre='vm+=11*mV').connect('i==j')
b2.Synapses(lIR, interlPPE, on_pre='vm+=11*mV').connect('i==j')
b2.Synapses(interhIRglobal, hIR, on_pre='vm-=11*mV').connect()
b2.Synapses(interhIRlocal, hIR, on_pre='vm-=11*mV').connect('i==j')
b2.Synapses(interlNPE, lNPE, on_pre='vm-=11*mV').connect('i==j')
b2.Synapses(interlPPE, lPPE, on_pre='vm-=11*mV').connect('i==j')

b2.run(10*b2.second)