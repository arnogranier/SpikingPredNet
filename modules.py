import brian2 as b2 
from neuron_model import neurons, synapses
        
class Area:
    def __init__(self, N, name, net, wINHIPE, wEXCIPE, wINHIIR=None, wEXCIIR=None, IRPoisson=False, recordspikes=False, onlyIR=False):
        self.net = net
        self.name = name
        self.N = N
        self.IRPoisson = IRPoisson
        if not IRPoisson:
            IR = neurons(N, name=name+'_IR', behavior='ir', net=net)
            interIR = neurons(N, name=name+'_interIR', behavior='i', net=net)
        else:
            IR = b2.NeuronGroup(N, 'rates : Hz', threshold='rand()<rates*dt',
                                name=name+'_IR', refractory=3*b2.ms)
            net.add(IR)
        if not IRPoisson:
            synapses(interIR, IR, 'i==j', wINHIIR, net)
            synapses(IR, IR, 'i==j', 35, net, delay=20*b2.ms)
        if not onlyIR:
            PPE = neurons(N, name=name+'_PPE', net=net)
            interPPE = neurons(N, name=name+'_interPPE', behavior='i', net=net)
            NPE = neurons(N, name=name+'_NPE', net=net)
            interNPE = neurons(N, name=name+'_interNPE', behavior='i', net=net)
            synapses(interPPE, PPE, 'i==j', wINHIPE, net)
            synapses(interNPE, NPE, 'i==j', wINHIPE, net)
            synapses(IR, interPPE, 'i==j', 35, net, delay=.1*b2.ms)
            synapses(IR, NPE, 'i==j', wEXCIPE, net, delay=.1*b2.ms)
            if not IRPoisson:
                synapses(PPE, IR, 'i==j', wEXCIIR, net)
                synapses(NPE, interIR, 'i==j', 35, net)

        if recordspikes:
            IRrecord = b2.SpikeMonitor(net[name+'_IR'], name=name+'_IR_RECORD')
            net.add(IRrecord)
            if not onlyIR:
                PPErecord = b2.SpikeMonitor(net[name+'_PPE'], name=name+'_PPE_RECORD')
                net.add(PPErecord)
                NPErecord = b2.SpikeMonitor(net[name+'_NPE'], name=name+'_NPE_RECORD')
                net.add(NPErecord)
        
    def set_rates(self, rate_var_name):
        self.net[self.name+'_IR'].run_regularly('rates = %s(t,i)'%rate_var_name, 1*b2.second)
    
    def record_variable(self, pop, name):
        record = b2.StateMonitor(self.net[self.name+'_'+pop], 'vm', record=True,
                                 name=self.name+'_'+name+'_RECORD')
        self.net.add(record)
        
    def __getitem__(self, key):
        return self.net[self.name+'_'+key+'_RECORD']
        
        
def connect(a1, a2, W, wEXCIPE, wEXCIIR=None, plastic=False):
    assert a1.net == a2.net 
    assert W.shape == (a1.N, a2.N)
    net = a1.net 
    sources, targets = W.nonzero()
    synapses(net[a1.name+'_IR'], net[a2.name+'_PPE'], (sources, targets), wEXCIPE, net)
    synapses(net[a1.name+'_IR'], net[a2.name+'_interNPE'], (sources, targets), 35, net)
    if not a1.IRPoisson:
        synapses(net[a2.name+'_PPE'], net[a1.name+'_interIR'], (targets, sources), 35, net)
        synapses(net[a2.name+'_NPE'], net[a1.name+'_IR'], (targets, sources), wEXCIIR, net)
    
    
    