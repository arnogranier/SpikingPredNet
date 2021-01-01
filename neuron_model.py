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
    """[summary]

    Args:
        n (int): [description]
        behavior (str, optional): [description]. Defaults to 'rs'.
        name (str, optional): [description]. Defaults to ''.
        net (b2.Network, optional): [description]. Defaults to None.
    
    Returns:
        [type]: [description]
    """
    
    eqs = '''
    dvm/dt = (gL*(EL - vm) + gL*DeltaT*exp((vm - VT)/DeltaT) + I - w)/(taum*C) : volt
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
    
    # Pick an electrophysiological behaviour
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
             w:float, net:b2.Network=None, lateralSTDP.bool=False,
             namesup:str='', wmax:float=35 **kwargs):
    """[summary]

    Args:
        p1 (str): [description]
        p2 (str): [description]
        motif ([type]): [description]
        w (float): [description]
        net (b2.Network, optional): [description]. Defaults to None.
        namesup (str, optional): [description]. Defaults to ''.

    Returns:
        [type]: [description]
    """
    
    if lateralSTDP:
        STDPmodel = '''w_syn : volt
                       thetaSTDP : volt
                       dapre/dt = -apre/taupre : 1 (event-driven)
                       dapost/dt = -apost/taupost : 1 (event-driven)'''
        STDPonpre = '''vm+=%s*int(w_syn>thetaSTDP)*mV
                       apre += Apre
                       w_syn = clip(w_syn+%s*apost*int(abs(apost)>.1)*mV,
                                    0*mV, %s*mV)''' % (wmax, wmax, wmax)
        STDPonpost = '''apost += Apost
                        w_syn = clip(w_syn+%s*apre*int(abs(apre)>.1)*mV,
                                     0*mV, %s*mV)''' % (wmax, wmax)
        s = b2.Synapses(net[p1.name], net[p2.name], model=STDPmodel,
                        on_pre=STDPonpre, on_post=STDPonpost, **kwargs,
                        name='s_%s_%s_%s'%(p1.name, p2.name, namesup))
    else:
        s = b2.Synapses(net[p1.name], net[p2.name], model='w_syn : volt',
                        on_pre='vm+=w_syn', **kwargs,
                        name='s_%s_%s'%(p1.name, p2.name)+namesup)
    if motif is None or motif == 'all':
        s.connect()
    elif isinstance(motif, str):
        s.connect(motif)
    elif (isinstance(motif, (list, tuple)) and len(motif)==2) \
      or (isinstance(motif, np.ndarray) and motif.shape[1] == 2):
        s.connect(i=motif[0], j=motif[1])
    s.w_syn = w*b2.mV
    if net is not None:
        net.add(s)
    return s