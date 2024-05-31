import qiskit
from qiskit.circuit.library import StatePreparation
from utils_sp import estimate_phases
import scipy
import numpy as np


def exact_moment(psi, hamil, k):
    return np.vdot(psi, scipy.linalg.expm(-1j * k * hamil) @ psi)

def moment(qc, L, J, g, eigenvalues_sort, k, c1, c2, 
    depolar=1e-3, shots=1e3, psi=None,
    trotterized_time_evolution=None,
    pi=None, lamb=None, h=None, hamil=None, beta=None
    ):
    t = c1 * k/2
    dt = t/np.ceil(t)
    nsteps = int(np.ceil(t))

    if psi is not None:
        qc = qiskit.QuantumCircuit(L+1, 1)
        statePrep_Gate = StatePreparation(np.kron(np.array([1,0]), psi), label=f'init')
        qc.reset([i for i in range(L+1)])
        qc.append(statePrep_Gate, [i for i in range(L+1)])
    
    print("t: ", t)
    print("dt: ", dt)
    print("nsteps: ", nsteps)
    rqc_layers = 7
    if dt < 0.1:
        rqc_layers = 3
    elif dt < 0.5:
        rqc_layers = 5
    

    ret = estimate_phases(L, J, g, qc, eigenvalues_sort, dt, 2,
                            shots, depolar, rqc_layers=rqc_layers,
                            rqc=trotterized_time_evolution is None, 
                            nsteps=nsteps, c2=c2, reuse_RQC=L-4,
                            trotterized_time_evolution=trotterized_time_evolution, 
                            pi=pi, lamb=lamb, h=h, hamil=hamil, beta=beta
                          )
    return ret[0][0][1]


def Fourier_coeffs(k, D, beta=5):
    if k == 0:
        return 0.5
    elif k%2 == 0:
        return 0
    elif k == D:
        j = (k-1)//2
        return -1j * np.sqrt(beta/(2*np.pi)) * np.exp(-beta) * (
            scipy.special.iv(j, beta)
        )/k
    else:
        j = (k-1)//2
        return -1j * np.sqrt(beta/(2*np.pi)) * np.exp(-beta) * (
            scipy.special.iv(j, beta) + scipy.special.iv(j+1, beta)
        )/k


def C(x, D, moments, beta=5):
    return np.sum(np.array(
           [Fourier_coeffs(k, beta) * np.exp(1j*x*k) * moments[i]
           for i, k in enumerate(range(-D, D+1))] 
    ))