import numpy as np
from deerlab import noiselevel, whitegaussnoise, dipolarkernel
from deerlab.dd_models import dd_gauss,dd_gauss2
from deerlab.bg_models import bg_exp


def test_filtered_movmean():
#============================================================
    "Check estimation of noiselevel using a moving-mean filter"

    np.random.seed(1)
    t = np.linspace(0,3,200)
    r = np.linspace(2,6,100)
    P = dd_gauss(r,[3, 0.5])
    lam = 0.25
    B = bg_exp(t,1.5,lam)
    noise = whitegaussnoise(t,0.03)
    V = dipolarkernel(t,r,lam,B)@P + noise


    truelevel = np.std(noise)
    approxlevel = noiselevel(V,'movmean')

    assert abs(approxlevel - truelevel) < 1e-2
#============================================================



def test_reference():
#============================================================
    "Check estimation of noiselevel using a reference signal"

    np.random.seed(1)
    t = np.linspace(0,3,200)
    r = np.linspace(2,6,100)
    P = dd_gauss(r,[3, 0.5])
    lam = 0.25
    B = bg_exp(t,1.5,lam)
    Vref = dipolarkernel(t,r,lam,B)@P
    noise = whitegaussnoise(t,0.03)
    V = Vref + noise


    truelevel = np.std(noise)
    approxlevel = noiselevel(V,Vref)

    assert abs(approxlevel - truelevel) < 1e-2
#============================================================


def test_filtered_savgol():
#============================================================
    "Check estimation of noiselevel using a Savitzky-Golay filter"

    np.random.seed(1)
    t = np.linspace(0,3,200)
    r = np.linspace(2,6,100)
    P = dd_gauss(r,[3, 0.5])
    lam = 0.25
    B = bg_exp(t,1.5,lam)
    noise = whitegaussnoise(t,0.03)
    V = dipolarkernel(t,r,lam,B)@P + noise

    truelevel = np.std(noise)
    approxlevel = noiselevel(V,'savgol')

    assert abs(approxlevel - truelevel) < 1e-2
#============================================================



def test_multiscan():
#============================================================
    "Check estimation of noiselevel using multiple scans of a signal"

    np.random.seed(1)
    t = np.linspace(0,5,300)
    r = np.linspace(2,6,200)
    P = dd_gauss(r,[4, 0.4])
    K = dipolarkernel(t,r)

    sigma_ref = 0.1
    N = 500
    V = np.zeros((len(t),N))
    for i in range(N):
        V[:,i] = K@P + whitegaussnoise(t,sigma_ref)

    sigma = noiselevel(V)

    assert abs(sigma - sigma_ref) < 1e-2
#============================================================


def test_complex():
#============================================================
    "Check estimation of noiselevel using a complex signal"

    np.random.seed(1)
    t = np.linspace(0,3,200)
    r = np.linspace(2,6,100)
    P = dd_gauss(r,[4, 0.4])
    K = dipolarkernel(t,r)
    lam = 0.25
    B = bg_exp(t,1.5,lam)

    noise = whitegaussnoise(t,0.03)
    V = dipolarkernel(t,r,lam,B)@P + noise
    Vco = V*np.exp(-1j*np.pi/5)

    truelevel = np.std(noise)
    approxlevel = noiselevel(Vco)

    assert abs(truelevel - approxlevel) < 1e-2
#============================================================