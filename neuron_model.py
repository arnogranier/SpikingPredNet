import brian2 as b2
import numpy as np 

# Parameters
C = 281 * b2.pF
gL = 30 * b2.nS
EL = -70.6 * b2.mV
VT = -50.4 * b2.mV
DeltaT = 2 * b2.mV
Vcut = VT + 5 * DeltaT
taupre = taupost = 20 * b2.ms 
Apre = .1
Apost = -.1

def neurons(n:int, behavior:str='pe', name:str='', net:b2.Network=None):
    """Create a brian2.NeuronGroup following the Adex neuron model
    dvm/dt = (gL*(EL - vm) + gL*DeltaT*exp((vm - VT)/DeltaT) + I - w)/(taum*C) 
    dw/dt = (a*(vm - EL) - w)/tauw 
    
    Args:
        n (int): Number of neurons
        behavior (str, optional): Electrophysiological behavior. Defaults to 
                                  'pe'
        name (str, optional): name if the population. Defaults to ''.
        net (b2.Network, optional): brian2.Network in which to add the
                                    population. Defaults to None.
    
    Returns:
        brian2.NeuronGroup: The population
    """
    
    eqs = '''
    dvm/dt = (gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT) + I - w)/(taum*C) : volt
    dw/dt = (a*(vm - EL) - w)/tauw : amp
    I : amp
    tauw : second
    a : siemens
    b : amp
    Vr : volt
    taum : 1
    '''

    group = b2.NeuronGroup(n, eqs, threshold='vm>Vcut', reset="vm=Vr; w+=b",
                           method='euler', name=name)
    group.vm = EL
    
    group.a = 4*b2.nS
    group.b = 0.0805*b2.nA
    group.Vr = -70.6*b2.mV
    
    if behavior == 'pe':
        group.tauw = 400*b2.ms
        group.taum = 6
    elif behavior == 'ir':
        group.tauw = 55*b2.ms
        group.taum = 3
    elif behavior == 'i':
        group.tauw = 10*b2.ms
        group.taum = 1
    else:
        raise NotImplementedError

    if net is not None:
        net.add(group)
    return group 

def synapses(p1:str, p2:str, motif:(None, str, list, tuple, np.ndarray),
             w:float, net:b2.Network, lateralSTDP:bool=False,
             namesup:str='', wmax:float=35, predSTDP:str=None,
             linkw:bool=False, W_syn:np.ndarray=None, **kwargs):
    """Add brian2.Synapses between population p1 and p2

    Args:
        p1 (str): Name of the projecting population
        p2 (str): Name of the receiving population
        motif (None, str, list, tuple, np.ndarray): Connection motif
        w (float): Weight of the synapses
        net (b2.Network, optional): brian2.Network containing populations p1 
                                    and p2 and to which we add the synapses.
                                    Defaults to None.
        lateralSTDP (bool, optional): boolean controlling weither the synapses
                                      follow the STDP learning rule for lateral
                                      synapses. Defaults to False.
        namesup (str, optional): Supplement of name because brian2 cannot use
                                 the same name twice, but we want to be able to
                                 add 2 synaptic complex from and to the same
                                 populations. Defaults to ''.
        wmax (float, optional): Max weight, usually sufficient to activate post
                                synaptic neuron with one presynaptic spike.
                                Defaults to 35**kwargs.
        predSTDP (str, optional): STDP learning rules for long range synapses
                                  Defaults to None.
        linkw (bool, optional): Are weights copied from another population?
                                Defaults to False.
        W_syn (np.ndarray, optional): intial weight matrix for learning
                                      predictions. Defaults to None.

    Returns:
        b2.Synapses: The synaptic complex 
    """
    
    if lateralSTDP:
        model = '''w_syn : volt
                   thetaSTDP : volt
                   dapre/dt = -apre/taupre : 1 (event-driven)
                   dapost/dt = -apost/taupost : 1 (event-driven)
                   wmax : volt (shared)'''
        onpre = '''vm+=wmax*int(w_syn>thetaSTDP)*mV
                   apre += Apre
                   w_syn = clip(w_syn+wmax*apost*int(apost<-.1), 0*mV, wmax)'''
        onpost = '''apost += Apost
                    w_syn = clip(w_syn+wmax*apre*int(apre>.1), 0*mV, wmax)'''
    elif predSTDP:
        model = '''w_syn : volt (shared) 
                   lastpost : second 
                   lastpre : second 
                   %s ''' % ('Wf:1' if not linkw else '')
        if predSTDP == '-+':
            onpre = '''vm+=w_syn*int(Wf>.5)
                       Wf = clip(Wf+0.01*int((t-lastpost)<20*ms), 0, 1)
                       lastpre = t'''
            onpost = '''lastpost = t'''
        elif predSTDP == '+-':
            onpre = '''vm+=w_syn*int(Wf>.5)
                       lastpre = t'''
            onpost = '''Wf = clip(Wf-0.01*int((t-lastpre)<1*ms), 0, 1)
                        lastpost = t'''
    else:
        model = 'w_syn : volt'
        onpre = 'vm+=w_syn'
        onpost = ''
    s = b2.Synapses(net[p1.name], net[p2.name], model==model,
                    on_pre=onpre, on_post=onpost, **kwargs,
                    name='s_%s_%s_%s'%(p1.name, p2.name, namesup))
    if motif is None or motif == 'all' or predSTDP:
        s.connect()
    elif isinstance(motif, str):
        s.connect(motif)
    elif (isinstance(motif, (list, tuple, np.ndarray)) and len(motif)==2):
        s.connect(i=motif[0], j=motif[1])
    else:
        raise NotImplementedError
    s.w_syn = w*b2.mV
    if predSTDP:
        s.lastpost = 0*b2.ms
        s.lastpre = 0*b2.ms
    elif lateralSTDP:
        s.wmax = wmax
    if net is not None:
        net.add(s)
    return s