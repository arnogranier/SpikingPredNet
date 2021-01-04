import brian2 as b2 
import numpy as np 

from neuron_model import neurons, synapses
        

class Area:
    """Main class representing a spiking predictive processing model of a 
    cortical area. Holds internal representation (IR) and prediction error (NPE 
    and PPE) neural populations, and all local connections between them.

    Args:
        N (int): Number of neurons
        name (str): Name of the area
        net (b2.Network): brian2.Network in which to add population and
                          synapses
        wINHIPE (float): Weight of synapses inhibiting PE populations
        wEXCIPE (float): Weight of synapses exciting PE populations
        wINHIIR (float, optional): Weight of synapses inhibiting the IR
                                   population. Defaults to 0..
        wEXCIIR (float, optional): Weight of synapses exciting the IR
                                   population. Defaults to 0..
        IRPoisson (bool, optional): If True then the IR population is defined 
                                    as following Poisson spike trains of rates
                                    defined by calling Area.set_rates. Defaults
                                    to False.
        IRSet (bool, optional): If True then the IR population is defined as
                                following deterministic spike trains defined by
                                calling Area.set_spikes. Defaults to False.
        recordspikes (bool, optional): If True then spike of the 3 populations
                                       are recorded through b2.SpikeMonitor,
                                       else not. Defaults to False.
        onlyIR (bool, optional): If True then skip the creation of PE
                                 populations. Defaults to False.
        lateralplasticity (bool, optional): If True then add lateral plastic
                                            synapses to the IR populations.
                                            Defaults to False.
        SetSpikes (list, optional): List of spikes to force in the IR
                                    population. The list should contain tuples
                                    (index of neuron, time of spike). Defaults
                                    to [].
        wmax (float, optional): Max weight. Defaults to 35..
        tointerneurons (bool, optional): If True then populations project back
                                         to their associated interneurons
                                         populations. Defaults to False.
    """
    
    def __init__(self, N:int, name:str, net:b2.Network, wINHIPE:float,
                 wEXCIPE:float, wINHIIR:float=0., wEXCIIR:float=0.,
                 IRPoisson:bool=False, IRSet:bool=False,
                 recordspikes:bool=False, onlyIR:bool=False,
                 lateralplasticity:bool=False, SetSpikes:list=[],
                 wmax:float=35., tointerneurons:bool=False):
        self.net = net
        self.name = name
        self.N = N
        self.IRPoisson = IRPoisson
        self.IRSet = IRSet
        
        # Poisson internal representations
        if IRPoisson:
            IR = b2.NeuronGroup(N, 'rates : Hz', threshold='rand()<rates*dt',
                                name=name+'_IR', refractory=3*b2.ms)
            net.add(IR)
            
        # Deterministic predefined spike trains internal representations
        elif IRSet:
            IR = b2.SpikeGeneratorGroup(N, [], []*b2.ms, name=name+'_IR')
            net.add(IR)
        
        # Freely evolving internal representations 
        else:
            IR = neurons(N, name=name+'_IR', behavior='ir', net=net)
        
        # Internal representation inhibitory interneurons
        interIR = neurons(N, name=name+'_interIR', behavior='i', net=net)
        
        # IR interneurons -> IR
        # IR -> IR
        if not (IRPoisson or IRSet):
            synapses(interIR, IR, 'i==j', wINHIIR, net)
            if not lateralplasticity:
                synapses(IR, IR, 'i==j', wmax, net, delay=20*b2.ms)
            else:
                synapses(IR, IR, 'i!=j', 1, net, lateralSTDP=True, 
                         delay=19.9*b2.ms)
        
        # IR -> IR interneurons (needed for FIG 2 F experiment)
        if tointerneurons:
            synapses(IR, interIR, 'i==j', wmax, net, delay=10*b2.ms)
        
        # Positive and negative prediction error populations 
        # PE interneurons -> PE
        # local IR -> PPE, NPE interneurons
        # PPE -> IR interneurons, NPE -> IR
        if not onlyIR:
            NPE = neurons(N, name=name+'_NPE', net=net)
            interNPE = neurons(N, name=name+'_interNPE', behavior='i', net=net)
            PPE = neurons(N, name=name+'_PPE', net=net)
            interPPE = neurons(N, name=name+'_interPPE', behavior='i', net=net)
            if tointerneurons:
                synapses(PPE, interPPE, 'i==j', wmax, net, delay=10*b2.ms)
            synapses(interNPE, NPE, 'i==j', wINHIPE, net)
            synapses(interPPE, PPE, 'i==j', wINHIPE, net)
            synapses(IR, interNPE, 'i==j', wmax, net)
            synapses(IR, PPE, 'i==j', wEXCIPE, net)
            if not (IRPoisson or IRSet):
                synapses(NPE, IR, 'i==j', wEXCIIR, net)
                synapses(PPE, interIR, 'i==j', wmax, net)
        
        # Recording of spikes
        if recordspikes:
            IRrecord = b2.SpikeMonitor(net[name+'_IR'], name=name+'_IR_RECORD')
            net.add(IRrecord)
            if not onlyIR:
                NPErecord = b2.SpikeMonitor(net[name+'_NPE'],
                                            name=name+'_NPE_RECORD')
                net.add(NPErecord)
                PPErecord = b2.SpikeMonitor(net[name+'_PPE'],
                                            name=name+'_PPE_RECORD')
                net.add(PPErecord)
        
        # Set spikes of neuron index at time time
        for (index, time) in SetSpikes:
            INITSG = b2.SpikeGeneratorGroup(1, [0], [time-.1,]*b2.second,
                                            name='INITSG')
            net.add(INITSG)
            synapses(INITSG, IR, (0, index), wmax, net)
        
    def set_rates(self, rate_var_name:str):
        """Set the rates of the Poisson internal representation population (if 
        it is indeed supposed to be Poisson).
        Since brian2genn does not support brian2.TimedArray as model variable,
        we pass the name of the brian2.TimedArray representing firing rates and
        call NeuronGroup.run_regularly to change firing rates every second. 

        Args:
            rate_var_name (str): Name of the brian2.TimedArray variable
        """
        assert self.IRPoisson
        IR = self.net[self.name+'_IR']
        IR.run_regularly('rates = %s(t,i)'%rate_var_name, 1*b2.second)
        
    def set_spikes(self, indices:(list, np.ndarray), times:(list, np.ndarray)):
        """Set the spikes of internal representations if these are set to be 
        a predefined deterministic spike train (brian2.SpikeGeneratorGroup)

        Args:
            indices (list, np.ndarray): indices of firing neurons
            times (list, np.ndarray): times of spikes
        """
        assert self.IRSet
        self.net[self.name+'_IR'].set_spikes(indices, times)    
    
    def record_variables(self, pop:str, name:str):
        """Record variables from population pop with a brian2.StateMonitor

        Args:
            pop (str): Name of the population to record from
            name (str): Name of the brian2.StateMonitor
        """
        record = b2.StateMonitor(self.net[self.name+'_'+pop], 'vm',
                                 record=True,
                                 name=self.name+'_'+name+'_RECORD')
        self.net.add(record)
        
    def __getitem__(self, key:str):
        """Get a Monitor with the Area[] syntax. SpikeMonitors are accessed
        through the keywords 'IR', 'NPE' and 'PPE'. StateMonitors are 
        accessed with their names.
        
        Args:
            key (str): Name of monitor

        Returns:
            brian2.SpikeMonitor or brian2.StateMonitor: The monitor
        """
        return self.net[self.name+'_'+key+'_RECORD']
        
        
def connect(a1:Area, a2:Area, W:np.ndarray, wEXCIPE:float, wEXCIIR:float=0.,
            plastic:bool=False, onlyPPE:bool=False, onlyNPE:bool=False,
            wmax:float=35, Wb:np.ndarray=None):
    """Connect two Area. a1 sends predictions to a2, and a2 sends back
    prediction errors to a1.

    Args:
        a1 (Area): Name of the higher population
        a2 (Area): Name of the lower population
        W (np.ndarray): prediction weight matrix
        wEXCIPE (float): Weight of synapses exciting lower PE populations
        wEXCIIR (float, optional): Weight of synapses exciting higher IR
                                   population. Defaults to 0..
        plastic (bool, optional): If True then prediction weights are plastic
                                  and learning of predictions occur. Defaults
                                  to False.
        onlyPPE (bool, optional): If True only connect negative PE population.
                                  Defaults to False.
        onlyNPE (bool, optional): If True only connect posistive PE population.
                                  Defaults to False.
        wmax (float, optional): Max weight. Defaults to 35.
        Wb (np.ndarray, optional): Backward weight matrix if we don't want to 
                                   initialize it as W. Defaults to None.
    """
    assert a1.net == a2.net 
    assert W.shape == (a1.N, a2.N)
    net = a1.net 
    
    # No need to build synapses with weight 0
    sources, targets = W.nonzero()
    
    # True if we want to learn prediction weights towards the 2 PE populations
    linkbool = not onlyPPE and not onlyNPE and plastic
    
    if not onlyPPE:
        
        # Higher IR -> NPE
        sWP = synapses(net[a1.name+'_IR'], net[a2.name+'_NPE'],
                      (sources, targets), wEXCIPE, net,
                      predSTDP='+-' if plastic else None)

        # Lower NPE -> IR interneurons
        if not (a1.IRPoisson or a1.IRSet):
            sM = synapses(net[a2.name+'_NPE'], net[a1.name+'_interIR'],
                          (targets, sources), wmax, net,
                          predSTDP='--' if plastic else None)
            
    if not onlyNPE:
        
        # Higher IR -> PPE interneurons
        sWN = synapses(net[a1.name+'_IR'], net[a2.name+'_interPPE'],
                 (sources, targets), wmax, net,
                 predSTDP='-+' if plastic else None, 
                 linkw=True)
        
        # In case of prediction weight learning, set weight matrices toward
        # NPE and PPE to be the same
        if linkw:
            sWN.variables.add_reference('Wf', sWP, 'Wf')
            
        # Lower PPE -> IR
        if not (a1.IRPoisson or a1.IRSet):
            synapses(net[a2.name+'_PPE'], net[a1.name+'_IR'],
                     (targets, sources), wEXCIIR, net,
                     predSTDP='++' if plastic else None)
            
    
    
    