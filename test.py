import brian2 as b2 
import numpy as np 


a = b2.SpikeGeneratorGroup(1, [0], [50,]*b2.ms)
b = b2.NeuronGroup(3, model='vm:volt')
c = b2.NeuronGroup(3, model='vm:volt')
s = b2.Synapses(a, b, model='Wf:1', on_pre='vm+=10*int(Wf>.5)*mV')