import brian2 as b2 
import brian2genn 
b2.set_device('genn')
import numpy as np
import matplotlib.pyplot as plt 
from neuron_model import C, gL, taum, EL, VT, DeltaT, Vcut
from neuron_model import neurons
from plots import raster_plot, rateplot2d
from area import Area 


net = b2.Network()
