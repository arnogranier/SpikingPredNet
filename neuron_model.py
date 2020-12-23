import brian2 as b2
import numpy as np 

# Parameters
C = 281 * b2.pF
gL = 30 * b2.nS
EL = -70.6 * b2.mV
VT = -50.4 * b2.mV
DeltaT = 2 * b2.mV
Vcut = VT + 5 * DeltaT

def neurons(n:int, behavior='rs', name='', net=None):
    
    eqs = """
    dvm/dt = (gL*(EL - vm) + gL*DeltaT*exp((vm - VT)/DeltaT) + I - w)/(taum*C) : volt
    dw/dt = (a*(vm - EL) - w)/tauw : amp
    I : amp
    tauw : second
    a : siemens
    b : amp
    Vr : volt
    taum : 1
    """

    group = b2.NeuronGroup(n, eqs, threshold='vm>Vcut',
                           reset="vm=Vr; w+=b", method='euler',
                           name=name)
    group.vm = EL
    
    # Pick an electrophysiological behaviour
    if behavior == 'rs':
        group.tauw = 400*b2.ms
        group.a = 4*b2.nS
        group.b = 0.0805*b2.nA
        group.Vr = -70.6*b2.mV
        group.taum = 6
    elif behavior == 'ir':
        group.tauw = 55*b2.ms
        group.a = 4*b2.nS
        group.b = 0.0805*b2.nA
        group.Vr = -70.6*b2.mV
        group.taum = 3
    elif behavior == 'i':
        group.tauw = 10*b2.ms
        group.a = 4*b2.nS
        group.b = 0.0805*b2.nA
        group.Vr = -70.6*b2.mV
        group.taum = 1
    elif behavior == 'fs':
        group.tauw = 144*b2.ms
        group.a = 2*C/(144*b2.ms)
        group.b = 0*b2.nA
        group.Vr = -70.6*b2.mV
        group.taum = 1
    else:
        raise NotImplementedError

    if net is not None:
        net.add(group)
    
    return group 


def synapses(p1, p2, motif, w, net=None, **kwargs):
    sinterPPEPPE = b2.Synapses(net[p1.name], net[p2.name], model='w_syn : volt',
                               on_pre='vm+=w_syn', **kwargs,
                               name='s_%s_%s'%(p1.name, p2.name))
    if motif is None or motif == 'all':
        sinterPPEPPE.connect()
    elif isinstance(motif, str):
        sinterPPEPPE.connect(motif)
    elif (isinstance(motif, (list, tuple)) and len(motif)==2) \
      or (isinstance(motif, np.ndarray) and motif.shape[1] == 2):
        sinterPPEPPE.connect(i=motif[0], j=motif[1])
    sinterPPEPPE.w_syn = w*b2.mV
    if net is not None:
        net.add(sinterPPEPPE)
    return sinterPPEPPE