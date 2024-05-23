import numpy as np
from pyqsp.completion import CompletionError
from pyqsp.angle_sequence import AngleFindingError
from utils_gsp import QETU, approx_polynomial, get_phis, QETU_cf, U_tau
from qiskit.quantum_info import state_fidelity
from matplotlib import pyplot as plt


def prepare_ground_state(initial_state, mu, d, c, phis_max_iter,
                         ground_state, L, J, g, ground_eigenvalue,
                         hamil, max_reps=5, tau=1, 
                         shift=0, a_max=1, fidelity_treshold=0.99):
    success_prob, end_state, poly, phis, layers, QETU_cf_mat  = get_success_prob(initial_state, mu, d, c, phis_max_iter, 
        max_reps, hamil, tau=tau, shift=shift, a_max=a_max, ground_state=ground_state)
    eigenvalue_estimate = 0
    print("\nF(a_max) = " + str(poly(a_max)**2))

    # This treshold is too low! We need to optimize the brickwall parameters for the non-periodic hamiltonian.
    if state_fidelity(end_state, ground_state) > fidelity_treshold:
        print("\n ---------- \n SUCCESS! \n")
        print("Fidelity of the initial state to the ground state: " + str(state_fidelity(initial_state, ground_state)))
        print("Fidelity of the prepared state to the ground state: " + str(state_fidelity(end_state, ground_state)))
        
        # TODO: Update the estimate_eig.
        #eigenvalue_estimate = estimate_eigenvalue(end_state, L, J, g)
        #print("\nEstimated Eigenvalue: ", eigenvalue_estimate)
        #print("Exact Eigenvalue: ", ground_eigenvalue)
        #print("Absolute Error: ", np.abs(ground_eigenvalue - eigenvalue_estimate))
    else:
        print("Fidelity of the prepared state to the ground state: " + str(state_fidelity(end_state, ground_state)))
        raise Exception("Badly configured for ground state preparation. Change (l, r) and try again!")
    return end_state, eigenvalue_estimate


def get_success_prob(state, x, d, c, max_iter_for_phis, reps,  hamil, steep = 0.01, tau=1, shift=0, a_max=0, ground_state=None):
    i = 0
    phis = []
    n = int(np.log(state.shape[0])/np.log(2)) - 1
    ket_0 = np.kron(np.array([1, 0]), np.array([1 if i==0 else 0 for i in range(2**n)]))
    a_list = np.linspace(-1, 1, 101)
    
    while True:
        try:
            phis = get_phis(x, d, steep, c=c)
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

    # Perform the Measurement
    end_state = state
    success_prob = 0
    layer = 0 
    
    QETU_cf_mat = QETU(hamil, phis[0], tau, shift)
    #QETU_cf_mat = QETU_cf(U_tau(hamil, tau), phis[0], c2=shift)

    for i in range(reps):
        layer = i+1
        if reps > 1:
            print("\nLayer " + str(i))
            
        meas_0 = np.kron(np.array([[1,0],[0,0]]), np.identity(2**n)) @ QETU_cf_mat
        end_state_0 = (meas_0 @ end_state)

        meas_1 = np.kron(np.array([[0,0],[0,1]]), np.identity(2**n)) @ QETU_cf_mat
        end_state_1 = (meas_1 @ end_state)

        prob_0 = np.linalg.norm(end_state_0)
        print("Prob 0: " + str(prob_0**2))

        prob_1 = np.linalg.norm(end_state_1)
        print("Prob 1: " + str(prob_1**2))

        success_prob = prob_0**2
        end_state = end_state_0 / prob_0
        poly = phis[2]

        if ground_state is not None:
            print(state_fidelity(end_state, ground_state))

        if a_max and np.abs(success_prob - poly(a_max)**2) < 1e-5:
            break
    return success_prob, end_state, poly, phis[0], layer, QETU_cf_mat

