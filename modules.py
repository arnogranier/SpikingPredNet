import brian2 as b2 
import numpy as np 

from neuron_model import neurons, synapses
        

class Area:
    """[summary]

    Args:
        N (int): [description]
        name (str): [description]
        net (b2.Network): [description]
        wINHIPE (float): [description]
        wEXCIPE (float): [description]
        wINHIIR (float, optional): [description]. Defaults to 0..
        wEXCIIR (float, optional): [description]. Defaults to 0..
        IRPoisson (bool, optional): [description]. Defaults to False.
        IRSet (bool, optional): [description]. Defaults to False.
        recordspikes (bool, optional): [description]. Defaults to False.
        onlyIR (bool, optional): [description]. Defaults to False.
        lateralplasticity (bool, optional): [description]. Defaults to False.
        SetSpikes (list, optional): [description]. Defaults to list().
        wmax (float, optional): [description]. Defaults to 35..
    """
    
    def __init__(self, N:int, name:str, net:b2.Network, wINHIPE:float,
                 wEXCIPE:float, wINHIIR:float=0., wEXCIIR:float=0.,
                 IRPoisson:bool=False, IRSet:bool=False,
                 recordspikes:bool=False, onlyIR:bool=False,
                 lateralplasticity:bool=False, SetSpikes:list=list(),
                 wmax:float=35.):
        self.net = net
        self.name = name
        self.N = N
        self.IRPoisson = IRPoisson
        self.IRSet = IRSet
        if IRPoisson:
            IR = b2.NeuronGroup(N, 'rates : Hz', threshold='rand()<rates*dt',
                                name=name+'_IR', refractory=3*b2.ms)
            net.add(IR)
        elif IRSet:
            IR = b2.SpikeGeneratorGroup(N, [], []*b2.ms, name=name+'_IR')
            net.add(IR)
        else:
            IR = neurons(N, name=name+'_IR', behavior='ir', net=net)
            interIR = neurons(N, name=name+'_interIR', behavior='i', net=net)
        if not (IRPoisson or IRSet):
            synapses(interIR, IR, 'i==j', wINHIIR, net)
            if not lateralplasticity:
                synapses(IR, IR, 'i==j', wmax, net, delay=20*b2.ms)
        if not onlyIR:
            PPE = neurons(N, name=name+'_PPE', net=net)
            interPPE = neurons(N, name=name+'_interPPE', behavior='i', net=net)
            NPE = neurons(N, name=name+'_NPE', net=net)
            interNPE = neurons(N, name=name+'_interNPE', behavior='i', net=net)
            synapses(interPPE, PPE, 'i==j', wINHIPE, net)
            synapses(interNPE, NPE, 'i==j', wINHIPE, net)
            synapses(IR, interPPE, 'i==j', wmax, net, delay=.1*b2.ms)
            synapses(IR, NPE, 'i==j', wEXCIPE, net, delay=.1*b2.ms)
            if not (IRPoisson or IRSet):
                synapses(PPE, IR, 'i==j', wEXCIIR, net)
                synapses(NPE, interIR, 'i==j', wmax, net)
        if lateralplasticity:
            synapses(IR, IR, 'i!=j', 1, net, lateralSTDP=True, namesup='P',
                     delay=19.9*b2.ms)
        if recordspikes:
            IRrecord = b2.SpikeMonitor(net[name+'_IR'], name=name+'_IR_RECORD')
            net.add(IRrecord)
            if not onlyIR:
                PPErecord = b2.SpikeMonitor(net[name+'_PPE'],
                                            name=name+'_PPE_RECORD')
                net.add(PPErecord)
                NPErecord = b2.SpikeMonitor(net[name+'_NPE'],
                                            name=name+'_NPE_RECORD')
                net.add(NPErecord)
        for (index, time) in SetSpikes:
            INITSG = b2.SpikeGeneratorGroup(1, [0], [time,]*b2.second,
                                            name='INITSG')
            net.add(INITSG)
            synapses(INITSG, IR, (0, index), wmax, net)
        
    def set_rates(self, rate_var_name:str):
        """[summary]

        Args:
            rate_var_name (str): [description]
        """
        assert self.IRPoisson
        IR = self.net[self.name+'_IR']
        IR.run_regularly('rates = %s(t,i)'%rate_var_name, 1*b2.second)
        
    def set_spikes(self, indices:(list, np.ndarray), times:(list, np.ndarray)):
        """[summary]

        Args:
            indices (list, np.ndarray): [description]
            times (list, np.ndarray): [description]
        """
        assert self.IRSet
        self.net[self.name+'_IR'].set_spikes(indices, times)    
    
    def record_variable(self, pop:str, name:str):
        """[summary]

        Args:
            pop (str): [description]
            name (str): [description]
        """
        record = b2.StateMonitor(self.net[self.name+'_'+pop], 'vm',
                                 record=True,
                                 name=self.name+'_'+name+'_RECORD')
        self.net.add(record)
        
    def __getitem__(self, key:str):
        """[summary]

        Args:
            key (str): [description]

        Returns:
            [type]: [description]
        """
        return self.net[self.name+'_'+key+'_RECORD']
        
        
def connect(a1:Area, a2:Area, W:np.ndarray, wEXCIPE:float, wEXCIIR:float=0.,
            plastic:bool=False, onlyNPE:bool=False, onlyPPE:bool=False,
            wmax:float=35):
    """[summary]

    Args:
        a1 (Area): [description]
        a2 (Area): [description]
        W (np.ndarray): [description]
        wEXCIPE (float): [description]
        wEXCIIR (float, optional): [description]. Defaults to 0..
        plastic (bool, optional): [description]. Defaults to False.
        onlyNPE (bool, optional): [description]. Defaults to False.
        onlyPPE (bool, optional): [description]. Defaults to False.
        wmax (float, optional): [description]. Defaults to 35.
    """
    assert a1.net == a2.net 
    assert W.shape == (a1.N, a2.N)
    net = a1.net 
    sources, targets = W.nonzero()
    if not onlyNPE:
        synapses(net[a1.name+'_IR'], net[a2.name+'_PPE'],
                 (sources, targets), wEXCIPE, net)
        if not (a1.IRPoisson or a1.IRSet):
            synapses(net[a2.name+'_PPE'], net[a1.name+'_interIR'],
                     (targets, sources), wmax, net)
    if not onlyPPE:
        synapses(net[a1.name+'_IR'], net[a2.name+'_interNPE'],
                 (sources, targets), wmax, net)
        if not (a1.IRPoisson or a1.IRSet):
            synapses(net[a2.name+'_NPE'], net[a1.name+'_IR'],
                     (targets, sources), wEXCIIR, net)
            
    
    
    