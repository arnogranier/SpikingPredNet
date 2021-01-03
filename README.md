# SpikingPredNet

This repository holds the code associated with my master thesis for the Neural Systems and Computation master at the Institute of Neuroinformatics, Zürich. <br>
The full master thesis __Predictive processing as a theory of cortical computation__ can be found [here](https://drive.google.com/file/d/10IlbAAJgDBCdhrmDhMqLsdihk7iuAW1R/view?usp=sharing). <br>
We propose here an implementation of a basic version of __predictive processing in a spiking neural network__, using the brian2 spiking neural networks simulator \[1\] and the brian2genn interface \[2\] to accelerate computation on GPUs.

# This is still under construction, final version should come around JAN 6

## Rationale
Master Thesis' Abstract:<br>
_In this work I discuss the question of understanding cortical computation and review what I think is the most promising approach of this question to date: the predictive processing framework. In this framework, the computational goal of the neocortex is free energy or prediction error minimization, and this goal is realized by hierarchical message passing and local computation in cortical microcircuitry. Many aspects of cognition have been found to be coherent with predictive processing, and the consistency of this theory with respect to experimental neuroscience data is actively investigated. The principles of the predictive processing framework, interpreted as a theory of cortical computation, would bridge the gap between the levels of neuronal activity and cognition or intelligence; and the construction of artificial systems following the same principles could lead to human- or mammal-like artificial intelligence. At the end of this work, I propose an embryo of implementation of these principles in a spiking neural network with learning and structure inspired from the neocortex._

The predictive processing framework offers a good starting point to think about cortical computation, both grounded in recent neuroscience and in theory capable of supporting cognition. Its implementation in a spiking neural network, proposed here, is a first step to further link the theory of predictive processing and computation in neural circuits. Moreover, predictive processing in spiking neural networks might prove directly and practically interesting for neuromorphic engineers as a powerful and general computational primitive using only local learning rules and approximating backpropagation. 

## How to use
You can use simple.py to reproduce experiments presented in section VII of the master thesis. 

Here we highlight how to use the modules.py, neuron_model.py modules to build spiking predictive processing models. 

First we need to import the needed libraries and modules:
```
import brian2 as b2 
import brian2genn 
import numpy as np 
import matplotlib.pyplot as plt 

from neuron_model import C, gL, EL, VT, DeltaT, Vcut
from neuron_model import neurons, synapses
from plots import rplots
from modules import Area, connect 
```

We can take advantage of the brian2genn interface to run simulations on the GPU:
```
b2.set_device('genn')
```

If we want to set the activity of one of the internal representation population
to a Poisson spike train, to experiment on how other components of the model
react, we need to degine a brian2.TimedArray corresponding to the firing rates
of this population:
```
activationLOW = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1]])
timedRatesLOW = b2.TimedArray(65*activationLOW*b2.Hz, dt=1*b2.second)
```

Next we create a brian2.Network to hold component of our simulation:
```
net = b2.Network()
```

We define model parameters, respectively weights inhibiting PE, exciting PE, inhibiting IR, exciting IR
(PE: prediction error populations, IR: internal representation populations)
```
wINHIPE, wEXCIPE = -25, 12
wINHIIR, wEXCIIR = -20, 25
```

Next we can define a prediction weight matrix encoding the weights from higher IR to lower PE. These weights can be learned, but in this simple example we simply set them to a fixed matrix W:
```
W = np.array([[1, 0, 1], 
              [0, 1, 1]])
```

We create spiking predictive processing areas (modelling cortical areas). This generally (except when onlyIR is true) creates 3 neural populations representing internal representations, negative and positive prediction errors:
```
LOW = Area(3, 'LOW', net, wINHIPE, wEXCIPE, IRPoisson=True, recordspikes=True)
HIGH = Area(2, 'HIGH', net, wINHIPE, wEXCIPE, wINHIIR, wEXCIIR, onlyIR=True, recordspikes=True)
```

We can set the firing rate of the lower population to be the previously defined Poisson spike train:
```
LOW.set_rates('timedRatesLOW')
```

The we connect the two areas. HIGH sends predictions to LOW, and LOW sends back prediction errors to HIGH:
```
connect(HIGH, LOW, W, wEXCIPE, wEXCIIR, plastic=False)
```

We finally run the simulation:
```
net.run(4*b2.second)
```

And we can easily plot raster plots using the plots.py module:
```
rplots(LOW['IR'], HIGH['IR'], LOW['PPE'], LOW['NPE'])
plt.show()
```

### References
\[1\] Marcel Stimberg, Romain Brette, and Dan FM Goodman. “Brian 2, an intuitive and efficient neural simulator”.Elife8 (2019), e47314.<br>
\[2\] Marcel Stimberg, Dan FM Goodman, and Thomas Nowotny. “Brian2GeNN: accelerating spiking neural network simulations with graphics hardware”.Scientific Reports 10.1 (2020), pp. 1–12.
