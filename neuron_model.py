import brian2 as b2

# Parameters
C = 281 * b2.pF
gL = 30 * b2.nS
taum = C / gL
EL = -70.6 * b2.mV
VT = -50.4 * b2.mV
DeltaT = 2 * b2.mV
Vcut = VT + 5 * DeltaT

def neurons(n:int, behavior='rs', name=''):
    
    eqs = """
    dvm/dt = (gL*(EL - vm) + gL*DeltaT*exp((vm - VT)/DeltaT) + I - w)/C : volt
    dw/dt = (a*(vm - EL) - w)/tauw : amp
    I : amp
    tauw : second
    a : siemens
    b : amp
    Vr : volt
    """

    group = b2.NeuronGroup(n, eqs, threshold='vm>Vcut',
                           reset="vm=Vr; w+=b", method='euler',
                           name=name)
    group.vm = EL
    
    # Pick an electrophysiological behaviour
    if behavior == 'rs':
        group.tauw = 144*b2.ms
        group.a = 4*b2.nS
        group.b = 0.0805*b2.nA
        group.Vr = -70.6*b2.mV
    elif behavior == 'b':
        group.tauw = 20*b2.ms
        group.a = 4*b2.nS
        group.b = 0.5*b2.nA
        group.Vr = VT+5*b2.mV 
    elif behavior == 'fs':
        group.tauw = 144*b2.ms
        group.a = 2*C/(144*b2.ms)
        group.b = 0*b2.nA
        group.Vr = -70.6*b2.mV
    else:
        raise NotImplementedError
    
    return group 
