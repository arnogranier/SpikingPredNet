import brian2 as b2 
import numpy as np 

a = b2.PoissonGroup(2, [50, 50]*b2.Hz)
b = b2.PoissonGroup(3, [50, 50, 50]*b2.Hz)

s1 = b2.Synapses(a, b, model='r:1', on_pre='''w=w+1''')
s2 = b2.Synapses(a, b, model='w:1', on_pre='''w=w+1''')
s1.variables.add_reference('w', s2, 'w')
s1.connect()
s2.connect()
b2.run(2*b2.second)
print(s1.w)