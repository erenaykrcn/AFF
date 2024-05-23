import qiskit
import scipy
import h5py
import os
import numpy as np
from pyqsp.completion import CompletionError
from pyqsp.angle_sequence import AngleFindingError
from qiskit.synthesis.two_qubit import TwoQubitBasisDecomposer
from qiskit.circuit.library import CXGate
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.providers.aer.noise import NoiseModel, errors, coherent_unitary_error
import rqcopt as oc
import qiskit.quantum_info as qi

from ground_state_prep import get_phis
from utils_gsp import construct_ising_local_term, approx_polynomial, get_phis, qc_U, qc_U_Strang, qc_QETU_cf_R, qc_QETU_R, QETU, QETU_heis_cf

import sys
sys.path.append("../../src/rqcopt")
from optimize import ising1d_dynamics_opt

decomp = TwoQubitBasisDecomposer(gate=CXGate())

# We demonstrate that we can implement the QETU Circuit with the TFIM Hamiltonian
# without including a controlled time evolution operator. This can be achieved by inserting controlled-
# single qubit gates, before and after the time evolution gate. To implement the conjugated time evolution
# operator, we insert X gates to the ancilla qubit, before and after the controlled-time evolution gate.


def get_error_from_sv(qc, qc_H, depolarizing_error, reps, L, J, g, ev, nshots=1e5, t1=3e10, t2=3e10, gate_t=100, coherent_error=0):
    backend = Aer.get_backend("aer_simulator")
    backend._configuration.max_shots = 1e10
    
    x1_error = errors.depolarizing_error(depolarizing_error*0.1, 1)
    x2_error = errors.depolarizing_error(depolarizing_error, 2)
    relax_error = errors.thermal_relaxation_error(t1, t2, gate_t)
    x1_error = x1_error.compose(relax_error)
    x2_error = x2_error.compose(relax_error)
    no_error = errors.depolarizing_error(0, L+1)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(x1_error, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(x2_error, ['cx', 'cy', 'cz', 'unitary'])
    noise_model.add_all_qubit_quantum_error(no_error, ['meas0'])
    noise_model.add_basis_gates(['unitary', 'meas0'])

    # Coherent overrotation around a random axis error.
    epsilon = coherent_error
    pauli = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    n = np.random.rand(3)
    n = n / np.linalg.norm(n)
    rot = np.zeros(2)
    for i, paul in enumerate(pauli):
        rot = rot + paul*n[i] 

    u_err_h =  scipy.linalg.expm(1j*epsilon*rot)
    noise_model.add_all_qubit_quantum_error(coherent_unitary_error(u_err_h), 'h')
    
    E_nps = []
    for i in range(reps):
        print("getting counts")
        counts_dict1 = execute(transpile(qc), backend, noise_model=noise_model, basis_gates=noise_model.basis_gates, shots=nshots).result().get_counts()
        counts_dict2 = execute(transpile(qc_H), backend, noise_model=noise_model, basis_gates=noise_model.basis_gates, shots=nshots).result().get_counts()
        print("gotten counts")
        
        E_np = estimate_eigenvalue_counts(counts_dict1, counts_dict2, L, J, g)
        E_nps.append(E_np)
        print(E_np)
    E_nps = np.array(E_nps)

    return np.linalg.norm(np.sum(E_nps)/E_nps.size - ev)


def prepare_ground_state_qiskit(L, J, g, t, mu, a_values, c2=0, d=30, c=0.95, steep=0.01, 
        max_iter_for_phis=10, repeat_qetu=3, RQC_layers=3, init_state=None, ground_state=None, reuse_RQC=False, eps=0):
    V_list = []
    dirname = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dirname, f"../../src/rqcopt/results/ising1d_L{reuse_RQC if reuse_RQC else L}_t{t}_dynamics_opt_layers{RQC_layers}.hdf5")

    try:
        with h5py.File(path, "r") as f:
            #assert f.attrs["L"] == L
            assert f.attrs["J"] == J
            assert f.attrs["g"] == g
            assert f.attrs["t"] == t
            V_list = list(f["Vlist"])
    except FileNotFoundError:
        strang = oc.SplittingMethod.suzuki(2, 1)
        _, coeffs_start_n5 = oc.merge_layers(2*strang.indices, 2*strang.coeffs)
        # divide by 2 since we are taking two steps
        coeffs_start_n5 = [0.5*c for c in coeffs_start_n5]
        print("coeffs_start_n5:", coeffs_start_n5)
        V_list = ising1d_dynamics_opt(5, t, False, coeffs_start_n5, path, niter=16)


    perms = [None if i % 2 == 0 else np.roll(range(L), -1) for i in range(len(V_list))]

    print("Running decomposition of two-qubit gates of the RQC Circuit...")
    qcs_rqc = []
    for V in V_list:
        #decomp = TwoQubitBasisDecomposer(gate=CXGate())
        #qc_rqc = decomp(V)
        qc_rqc = qiskit.QuantumCircuit(2)
        qc_rqc.unitary(V, [0, 1])
        qcs_rqc.append(qc_rqc)

    backend = Aer.get_backend("statevector_simulator")    

    phis = []
    i = 0
    while True:
        try:
            phis = get_phis(mu, d, steep, c=c)
            break
        except CompletionError:
            print("Completion Error encountered!")
            if i>max_iter_for_phis:
                raise Exception("Max Iteration for estimating the phis breached!")
            i = i + 1
            c = c - 0.01
            print(f"c updated to {c}!")
            if i > max_iter_for_phis / 2:
                print(f"QSP did not work for d = {d}, updating d to {d-4}")
                d = d - 4
        except AngleFindingError:
            print("AngleFindingError encountered!")
            if i>max_iter_for_phis:
                raise Exception("Max Iteration for estimating the phis breached!")
            i = i + 1
            
            if i > max_iter_for_phis / 2:
                print(f"QSP did not work for d = {d}, updating d to {d-4}")
                d = d - 4
    phis_list = [phi+eps for phi in phis[0]]
    
    print("F(a_max)^2: ", phis[2](a_values[0])**2)

    qc_U_ins = qc_U(qcs_rqc, L, perms)
    qc_QETU = qc_QETU_cf_R(qc_U_ins, phis_list, c2)
    qc_ins = qiskit.QuantumCircuit(L+1, L+1)

    statevectors_bR = []
    statevectors_aR = []

    if init_state is not None:
        qc_ins.initialize(init_state)
    for i in range(repeat_qetu):
        print("Layer ", i)
        qc_ins.append(qc_QETU.to_gate(), [i for i in range(L+1)])
        bR = execute(transpile(qc_ins), backend).result().get_statevector().data
        statevectors_bR.append(bR)
        # TODO: This workaround should be only temporary! Reset is not working as expected!
        #qc_ins.reset(L)
        #aR = execute(transpile(qc_ins), backend).result().get_statevector()
        aR = np.kron(np.array([[1,0],[0,0]]), np.identity(2**L)) @ bR
        aR = aR / np.linalg.norm(aR)
        statevectors_aR.append(aR)
        statePrep_Gate = StatePreparation(aR, label='meas0')
        qc_ins.reset([i for i in range(L+1)])
        qc_ins.append(statePrep_Gate, [i for i in range(L+1)])

    qc_ins2 = qc_ins.copy()
    qc_ins2.h([i for i in range(L)])
    qc_ins.measure([i for i in range(L+1)], [i for i in range(L+1)])
    qc_ins2.measure([i for i in range(L+1)], [i for i in range(L+1)])

    qc_RQC = qc_ins.copy()
    qc_H_RQC = qc_ins2.copy()

    backend = Aer.get_backend("statevector_simulator")
    qc_U_ins = qc_U_Strang(L, J, g, t, 1)
    qc_QETU = qc_QETU_cf_R(qc_U_ins, phis_list, c2)
    qc_ins = qiskit.QuantumCircuit(L+1, L+1)

    if init_state is not None:
        qc_ins.initialize(init_state)
    for i in range(repeat_qetu):
        print("Layer ", i)
        qc_ins.append(qc_QETU.to_gate(), [i for i in range(L+1)])
        bR = execute(transpile(qc_ins), backend).result().get_statevector().data
        statevectors_bR.append(bR)
        aR = np.kron(np.array([[1,0],[0,0]]), np.identity(2**L)) @ bR
        aR = aR / np.linalg.norm(aR)
        statevectors_aR.append(aR)
        statePrep_Gate = StatePreparation(aR, label='meas0')
        qc_ins.reset([i for i in range(L+1)])
        qc_ins.append(statePrep_Gate, [i for i in range(L+1)])
        #qc_ins.reset(L)

    qc_ins2 = qc_ins.copy()
    qc_ins2.h([i for i in range(L)])
    qc_ins.measure([i for i in range(L+1)], [i for i in range(L+1)])
    qc_ins2.measure([i for i in range(L+1)], [i for i in range(L+1)])

    qc_STR = qc_ins.copy()
    qc_H_STR = qc_ins2.copy()
    return qc_RQC, qc_H_RQC, qc_STR, qc_H_STR



def Pr_counts(counts, j, zj, *zjplus1):
    # Helper Function to retrieve the probability of measuring the qubit number j (and optionally j+1)
    prob = 0
    nshots = 0
    for key, value in counts.items():
        nshots += value
    
    for bitstring, count in counts.items():
        bitstring = bitstring[::-1]
        bitstring = bitstring[:-1]
        
        L = len(bitstring)
        
        if not zjplus1:
            if bitstring[j] == str(zj):
                prob += count/nshots
        else:
            if j == L-1:
                if bitstring[j] == str(zj) and bitstring[0] == str(zjplus1[0]):
                    prob += count/nshots
            else:
                if bitstring[j] == str(zj) and bitstring[j+1] == str(zjplus1[0]):
                    prob += count/nshots
    return prob


def estimate_eigenvalue_counts(counts1, counts2, L, J, g):
    E1 = 0
    # For consistency with qib, upper range has to be L!
    for j in range(L):
        for zj in range(2):
            for zjplus1 in range(2):
                E1 = E1 + (((-1)**(zj + zjplus1)) * Pr_counts(counts1, j, zj, zjplus1))
    
    E2 = 0
    for j in range(L):
        for zj in range(2):
            E2 = E2 + (((-1)**zj) * Pr_counts(counts2, j, zj))

    return J*E1 + g*E2



def qetu_rqc_oneLayer(L, J, g, t, mu, a_values, c2=0, d=30, c=0.95, 
    steep=0.01, max_iter_for_phis=10, RQC_layers=5, 
    init_state=None, split_U=1, reuse_RQC=0, qc_U_custom=None,
    custom_qc_QETU_cf_R=None, qc_cU_custom=None, hamil=None,
    H1=None, H2=None, heis_c2=0, eps=0, lambda_est=None, reverse=False
    ):
    # One QETU Layer for TFIM Hamiltonian. You can also custom give the time
    # evolution block and the control-free implementation of QETU.

    t = t/split_U
    print("dt: ", t)
    V_list = []
    dirname = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dirname, f"../../src/rqcopt/results/ising1d_L{reuse_RQC if reuse_RQC else L}_t{t}_dynamics_opt_layers{RQC_layers}.hdf5")

    if qc_U_custom is None and qc_cU_custom is None:	
        try:
            with h5py.File(path, "r") as f:
                #assert f.attrs["L"] == L
                assert f.attrs["J"] == J
                assert f.attrs["g"] == g
                assert f.attrs["t"] == t
                V_list = list(f["Vlist"])
        except FileNotFoundError:
            strang = oc.SplittingMethod.suzuki(2, 1)
            _, coeffs_start_n5 = oc.merge_layers(2*strang.indices, 2*strang.coeffs)
            # divide by 2 since we are taking two steps
            coeffs_start_n5 = [0.5*c for c in coeffs_start_n5]
            print("coeffs_start_n5:", coeffs_start_n5)
            V_list = ising1d_dynamics_opt(5, t, False, coeffs_start_n5, path, niter=16)
        
            if RQC_layers >= 7:
                print("optimizing RQC for 7 layers")
                V_list = ising1d_dynamics_opt(7, t, True, niter=200)
            if RQC_layers == 9:
                print("optimizing RQC for 9 layers")
                V_list = ising1d_dynamics_opt(9, t, True, niter=200, tcg_abstol=1e-12, tcg_reltol=1e-10)



    perms = [None if i % 2 == 0 else np.roll(range(L), -1) for i in range(len(V_list))]

    print("Running decomposition of two-qubit gates of the RQC Circuit...")
    qcs_rqc = []
    for V in V_list:
        #decomp = TwoQubitBasisDecomposer(gate=CXGate())
        #qc_rqc = decomp(V)
        qc_rqc = qiskit.QuantumCircuit(2)
        qc_rqc.unitary(V, [0, 1])
        qcs_rqc.append(qc_rqc)

    backend = Aer.get_backend("statevector_simulator")
    phis = []
    i = 0
    while True:
        try:
            phis = get_phis(mu, d, steep, c=c, reverse=reverse)
            break
        except CompletionError:
            print("Completion Error encountered!")
            if i>max_iter_for_phis:
                raise Exception("Max Iteration for estimating the phis breached!")
            i = i + 1
            c = c - 0.01
            print(f"c updated to {c}!")
            if i > max_iter_for_phis / 2:
                print(f"QSP did not work for d = {d}, updating d to {d-4}")
                d = d - 4
        except AngleFindingError:
            print("AngleFindingError encountered!")
            if i>max_iter_for_phis:
                raise Exception("Max Iteration for estimating the phis breached!")
            i = i + 1
            
            if i > max_iter_for_phis / 2:
                print(f"QSP did not work for d = {d}, updating d to {d-4}")
                d = d - 4
    phis_list = [phi+eps for phi in phis[0]]

    print("F(a_max)^2: ", phis[2](a_values[0])**2)
    print("F(a_premax)^2: ", phis[2](a_values[1])**2)
    print("F(x)^2: ", phis[2](mu)**2)

    qc_U_ins = qiskit.QuantumCircuit(L)
    if qc_U_custom is None and lambda_est is None:
        for i in range(split_U):
            qc_U_ins.append(qc_U(qcs_rqc, L, perms).to_gate(), [i for i in range(L)])
    elif lambda_est is not None:
        qc_Phase = qiskit.QuantumCircuit(L)
        for i in range(L):
            qc_Phase.p(-t * split_U * lambda_est, i)
            qc_Phase.x(i)
            qc_Phase.p(-t * split_U * lambda_est, i)
            qc_Phase.x(i)
        qc_U_ins.append(qc_Phase.to_gate(), [j for j in range(L)])
    else:
        qc_U_ins = qc_U_custom
    
    if custom_qc_QETU_cf_R is not None:
        qc_QETU = custom_qc_QETU_cf_R(qc_U_ins, phis_list, c2, split_U=split_U)
    elif qc_cU_custom is not None:
        qc_QETU = qc_QETU_R(qc_cU_custom, phis_list, c2)
    else:
        qc_QETU = qc_QETU_cf_R(qc_U_ins, phis_list, c2)

    qc_ins = qiskit.QuantumCircuit(L+1)

    QETU_mat = None
    if hamil is not None and H1 is None:
        backend = Aer.get_backend("unitary_simulator")
        qc_unit = execute(transpile(qc_QETU), backend).result().get_unitary(qc_QETU, L+1).data
        QETU_mat = QETU(hamil, phis_list, t, c2)
        print("QETU Error: ", np.linalg.norm(qc_unit-QETU_mat, ord=2))

    if H1 is not None and H2 is not None:
        backend = Aer.get_backend("unitary_simulator")
        QETU_mat = QETU_heis_cf(H1, H2, phis_list, t, heis_c2, 10)
        qc_unit = execute(transpile(qc_QETU), backend).result().get_unitary(qc_QETU, L+1).data
        print("QETU Error: ", np.linalg.norm(qc_unit-QETU_mat, ord=2))


    if init_state is not None:
        qc_ins.initialize(init_state)
    qc_ins.append(qc_QETU.to_gate(), [i for i in range(L+1)])


    return qc_ins, (phis_list, phis[1], phis[2]), QETU_mat
