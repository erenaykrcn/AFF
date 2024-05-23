import qiskit
import numpy as np
from scipy.linalg import expm
import cvxpy as cp
from numpy import polynomial as P
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
import rqcopt as oc
import scipy
from qiskit.transpiler.passes import GatesInBasis
from qiskit.converters import circuit_to_dag
from qiskit import QuantumCircuit

# Below are the functions to randomize a diagonal hamiltonian
# and QETU circuit using directly controlled time evolution block.
# Implemented with numpy.

unnorm_Hadamard = np.array([[1, 1],[1, -1]])
ket_0 = np.array([1, 0])

def H(a, dim):
    if dim == 0:
        return 2*np.arccos(a)
    H = np.identity(dim)
    H[0][0] = 2*np.arccos(a)
    
    ai = 0.1
    for i in range(1, dim):
        ai = ai - 1e-8
        H[i][i] = 2*np.arccos(ai)
    return H

def get_a(a_max_arg=None, a_premax_arg=None, cos_eta_2=0.6):
    if a_max_arg:
        a_max = a_max_arg
    else:
        a_max = random() * 0.3 + cos_eta_2
    
    if a_premax_arg:
        a_premax = a_premax_arg
    else:
        a_premax = random()*0.05

    print("a_premax: " + str(a_premax))
    print("a_max: " + str(a_max))
    return a_max, a_premax

def S(phi, dim):
    S_z = np.array([[np.exp(1j*phi), 0],[0, np.exp(-1j*phi)]])
    S_x =  0.5*(unnorm_Hadamard @ S_z @ unnorm_Hadamard)
    return np.kron(S_x, np.identity(dim))

def W_z(a, dag, dim=1):
    # Signal Operator
    theta = np.arccos(a)
    if dag:
        theta = -theta
    Z = np.array([[1,0],[0,-1]])
    Z_exp = np.identity(1)
    for i in range(dim):
        Z_exp = np.kron(Z_exp, Z)
    return scipy.linalg.expm(1j*theta*Z_exp)

def cU(H, dagger, tau=1, shift=0):
    offset = len(H)
    dim = 2*offset
    cU  = (1+0j) * np.identity(dim)
    
    H = tau*H + shift*np.identity(offset)
    
    expH = expm(-1j * H)
    if dagger:
        expH = expH.transpose().conjugate()
        
    for i in range(offset):
        for j in range(offset):
            cU[offset+i][offset+j] = expH[i][j]
    return cU

def QETU(H, phis, tau=1, shift=0):
    U = S(phis[len(phis)-1], len(H))
    for i in range(1, len(phis)):
        U = U @ cU(H, i % 2 == 0, tau, shift) @ S(phis[len(phis)-1-i], len(H))
    return U


# Helper functions for visualizing.
def W_x(a):
    return np.array([[a, 1j*np.sqrt(1-a**2)], [1j*np.sqrt(1-a**2), a]])

def S_1(phi):
    return np.array([[np.exp(1j*phi), 0],[0, np.exp(-1j*phi)]])

def U(phi, a):
    U = S_1(phi[0])
    for i in range(1, len(phi)):
        U = U @ W_x(a) @ S_1(phi[i])
    return U


def construct_ising_local_term(J, g):
    """
    Construct local interaction term of Ising Hamiltonian on a one-dimensional
    lattice for interaction parameter `J` and external field parameter `g`.
    """
    # Pauli-X and Z matrices
    X = np.array([[0.,  1.], [1.,  0.]])
    Z = np.array([[1.,  0.], [0., -1.]])
    I = np.identity(2)
    return J*np.kron(Z, Z) + g*0.5*(np.kron(X, I) + np.kron(I, X))


# Quantum Signal Processing step to optimize the phi values.
def get_phis(x, d, h, c=0.99, reverse=False):
    assert d % 2 == 0
    a_list = np.linspace(-1, 1, 101)
    
    poly_even_step, even_coeffs = approx_polynomial(500, d, x-h, x+h, c, 0.01, reverse=reverse)
    poly_even_step  = poly_even_step.convert(kind=P.Polynomial)
    
    ket_0 = np.array([1,0])
    
    # TODO: Degrees of freedom is actually d/2! With that we can get much smoother functions!
    phi_primes    = QuantumSignalProcessingPhases(poly_even_step, signal_operator="Wx")
    phi_primes[0] = (phi_primes[-1] + phi_primes[0])/2
    phi_primes[-1] = phi_primes[0]
    
    """qsp_polynom = [poly_even_step(a) for a in a_list]
    plt.plot(a_list, qsp_polynom, label="Polynom")
    qsp_polynom = [np.vdot(ket_0, U(phi_primes, a)@ket_0).real for a in a_list]
    plt.plot(a_list, qsp_polynom, "--", label=r"$Re{\langle0|U_{\phi}|0\rangle}$")
    qsp_polynom = [np.vdot(ket_0, U(phi_primes, a)@ket_0).imag for a in a_list]
    plt.plot(a_list, qsp_polynom, "--", label=r"$Im{\langle0|U_{\phi}|0\rangle}$")
    plt.legend()"""
    
    phis = [phi_prime + np.pi/4 if i==0 or i==len(phi_primes)-1 else phi_prime + np.pi/2 for i, phi_prime in enumerate(phi_primes)]
    return (phis, phi_primes, poly_even_step)


# Functions for approximating the even step function with polynomials.
def Cheby_polyf(x: [float], k: int):
    x = np.array(x)
    y = np.arccos(x)
    ret = np.cos((2*k)*y)
    return ret


def approx_polynomial(M, d, sigma_minus, sigma_plus, c, eps, reverse=False,  x_list_arg=None):
    assert d%2 == 0
    x = []
    for j in range(M):
        xj = -np.cos(j*np.pi/(M-1))
        if np.abs(xj)<= sigma_minus or np.abs(xj) >= sigma_plus:
            x.append(xj)
    
    def cost_func(x_c, coeff, c):
        A = []
        for k in range(int(coeff.shape[0])):
            A.append(Cheby_polyf(x_c, k))
        A = np.array(A)
        A = A.transpose()
        
        b = [0 for _ in range(len(x_c))]
        for j, xj in enumerate(x_c):
            if reverse:
                if np.abs(xj) >= sigma_minus:
                    b[j] = 0
                else:
                    b[j] = c
            else:
                if np.abs(xj) <= sigma_minus:
                    b[j] = 0
                else:
                    b[j] = c
        b = np.array(b)
        return (A @ coeff) - b
    
    x_list = np.linspace(-1, 1, 101)
    if x_list_arg:
        x_list = x_list_arg
    
    coeff = cp.Variable(int(d/2))
    constraints = [np.sum([ck*Cheby_polyf([x_i], k)[0] for k,ck in enumerate(coeff)]) <= c-eps for x_i in x_list]
    constraints += [np.sum([ck*Cheby_polyf([x_i], k)[0] for k,ck in enumerate(coeff)]) >= eps for x_i in x_list]

    objective = cp.Minimize(cp.sum_squares(cost_func(x, coeff, c)))
    
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    coeffs = coeff.value

    func = [(  np.sum([ck*Cheby_polyf([x], k)[0] for k,ck in enumerate(coeffs)])  ) for x in x_list]
    for f in func:
        if f < 0 or f > 1: 
            raise Exception("Found polynomial exceeds the [0,1] Interval!")
    
    return P.chebyshev.Chebyshev([coeffs[int(i/2)] if i%2==0 else 0 for i in range(2*len(coeffs)-1)]), coeffs


def qc_U(two_qubit_gates, L, perms):
    U = qiskit.QuantumCircuit(L)
    for layer, qc_gate in enumerate(two_qubit_gates):
        assert L%2 == 0
        for j in range(L//2):
            if perms[layer] is not None:
                U.append(qc_gate.to_gate(), [perms[layer][2*j], perms[layer][2*j+1]])
            else:
                U.append(qc_gate.to_gate(), [2*j, 2*j+1])
    return U


def qc_U_Strang(L, J, g, t, nsteps):
    U = qiskit.QuantumCircuit(L)
    
    dt = t/nsteps
    hloc = construct_ising_local_term(J, g)
    coeffs = oc.SplittingMethod.suzuki(2, 1).coeffs
    #strang = oc.SplittingMethod.suzuki(2, 1)
    #_, coeffs = oc.merge_layers(2*strang.indices, 2*strang.coeffs)
    #coeffs = [0.5*c for c in coeffs]

    Vlist = [scipy.linalg.expm(-1j*c*dt*hloc) for c in coeffs]
    Vlist_gates = []
    for V in Vlist:
        #decomp = TwoQubitBasisDecomposer(gate=CXGate())
        #qc = decomp(V)
        qc = qiskit.QuantumCircuit(2)
        qc.unitary(V, [0, 1], label='str')
        Vlist_gates.append(qc)
    perms = [None if i % 2 == 0 else np.roll(range(L), -1) for i in range(len(coeffs))]

    for layer, qc_gate in enumerate(Vlist_gates):
        assert L%2 == 0
        for j in range(L//2):
            if perms[layer] is not None:
                U.append(qc_gate.to_gate(), [perms[layer][2*j], perms[layer][2*j+1]])
            else:
                U.append(qc_gate.to_gate(), [2*j, 2*j+1])
    return U


unnorm_Hadamard = np.array([[1, 1],[1, -1]])


def U_tau(H, c1=1):
    # U will be delivered in the further steps by the Riemannian optimization. 
    # For now we take it to be a black-box. The default tau is also 1/2 as the
    # control free implementation rescales the evolution time.
    expH = expm(-1j * 0.5 * c1 * H )
    return expH


def cfree_shifted_U(U_sh, dagger, c2=0):
    L = int(np.log(len(U_sh)) / np.log(2))
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    I = np.identity(2)
    
    X_I = X
    for j in range(L):
        X_I = np.kron(X_I, I)
        
    K = np.identity(1)
    for j in range(L):
        if j % 2 == 0:
            K = np.kron(K, Y)
        else:
            K = np.kron(K, Z)
    
    # K is controlled by the ancilla qubit, applied when qubit is 0.
    cK  = (1+0j) * np.identity(2*len(U_sh))
    offset = len(U_sh)
    for i in range(offset):
        for j in range(offset):
            cK[i][j] = K[i][j]
    
    cU = (1+0j) * np.identity(2*len(U_sh))
    U_ext = np.kron(I, U_sh)
    cU = cK @ U_ext @ cK
    
    # Shift the global phase, controlled by the ancilla.
    # TODO: Learn if this is physically realizable.
    cC2 = (1+0j) * np.identity(2*len(U_sh))
    C2 = expm(-1j*c2*np.identity(offset))
    for i in range(offset):
        for j in range(offset):
            cC2[offset+i][offset+j] = C2[i][j]
    cU = cC2 @ cU
    
    if dagger:
        cU = X_I @ cU @ X_I
    return cU


def QETU_cf(U, phis, c2=0):
    Q = S(phis[len(phis)-1], len(U))
    for i in range(1, len(phis)):
        Q = Q @ cfree_shifted_U(U, i % 2 == 1, c2) @ S(phis[len(phis)-1-i], len(U))
    return Q


def qc_cfU_R(qc_U, dagger, c2):
    L = qc_U.num_qubits
    qc_cfU = qiskit.QuantumCircuit(L+1)
    
    if dagger:
        qc_cfU.x(L)
    
    qc_cfU.x(L)
    for j in range(L-1, -1, -1):
        if j % 2 == 1:
            qc_cfU.cy(L, j)
        else:
            qc_cfU.cz(L, j)
    qc_cfU.x(L)
    
    qc_cfU.append(qc_U.to_gate(), [i for i in range(L)])
    
    qc_cfU.x(L)
    for j in range(L-1, -1, -1):
        if j % 2 == 1:
            qc_cfU.cy(L, j)
        else:
            qc_cfU.cz(L, j)
    qc_cfU.x(L)

    qc_cfU.cp(-c2, L, 0)
    qc_cfU.x(0)
    qc_cfU.cp(-c2, L, 0)
    qc_cfU.x(0)
    
    if dagger:
        qc_cfU.x(L)
    
    return qc_cfU


def qc_QETU_cf_R(qc_U, phis, c2=0):
    """
        Control Free Implementation of the QETU Circuit
        for the TFIM Hamiltonian. Encoded reversely as 
        qiskit uses Little Endian.
    """
    L = qc_U.num_qubits
    qc = qiskit.QuantumCircuit(L+1)
    qc.rx(-2*phis[0], L)
    for i in range(1, len(phis)):
        qc.append(qc_cfU_R(qc_U, i%2==0, c2).to_gate(), [i for i in range(L+1)])
        qc.rx(-2*phis[i], L)
    return qc


def qc_QETU_R(qc_cU, phis, c2=0):
    """
        Control Free Implementation of the QETU Circuit
        Encoded reversely as qiskit uses Little Endian.
    """
    L = qc_cU[0].num_qubits-1
    qc = qiskit.QuantumCircuit(L+1)
    qc.rx(-2*phis[0], L)
    for i in range(1, len(phis)):
        if i%2==1:
            qc.append(qc_cU[0].to_gate(), [i for i in range(L+1)])
        else:
            qc.append(qc_cU[1].to_gate(), [i for i in range(L+1)])
        qc.cp(-c2, L, 0)
        qc.x(0)
        qc.cp(-c2, L, 0)
        qc.x(0)
        qc.rx(-2*phis[i], L)
    return qc


def qc_QETU_cf_heis_R(qc_U, phis, c2=0, split_U=1):
    """
        Control Free Implementation of the QETU Circuit
        for the Heisenberg Hamiltonian. Encoded 
        reversely as qiskit uses Little Endian.
    """
    qc_U1, qc_U2 = qc_U

    L = qc_U1.num_qubits
    qc = qiskit.QuantumCircuit(L+1)
    qc.rx(-2*phis[0], L)

    for i in range(1, len(phis)):
        qc_cfU_heis_R = qiskit.QuantumCircuit(L+1)
        if i%2==0:
            qc_cfU_heis_R.x(L)
        
        #U1 ------------
        # control
        qc_cfU_heis_R.x(L)
        for j in range(L-1, -1, -1):
            if j % 2 == 1:
                qc_cfU_heis_R.cz(L, j)
        qc_cfU_heis_R.x(L)
        #control
        
        for l in range(split_U):
            qc_cfU_heis_R.append(qc_U1.to_gate(), [i for i in range(L)])
        
        #control
        qc_cfU_heis_R.x(L)
        for j in range(L-1, -1, -1):
            if j % 2 == 1:
                qc_cfU_heis_R.cz(L, j)
        qc_cfU_heis_R.x(L)
        #control

        #U2 ------------
        # control
        qc_cfU_heis_R.x(L)
        for j in range(L-1, -1, -1):
            if j % 2 == 1:
                qc_cfU_heis_R.cx(L, j)
        qc_cfU_heis_R.x(L)
        #control
        
        for l in range(split_U):
            qc_cfU_heis_R.append(qc_U2.to_gate(), [i for i in range(L)])
        
        
        #control
        qc_cfU_heis_R.x(L)
        for j in range(L-1, -1, -1):
            if j % 2 == 1:
                qc_cfU_heis_R.cx(L, j)
        qc_cfU_heis_R.x(L)
        #control
        #-----------------

        qc_cfU_heis_R.cp(-c2, L, 0)
        qc_cfU_heis_R.x(0)
        qc_cfU_heis_R.cp(-c2, L, 0)
        qc_cfU_heis_R.x(0)

        if i%2==0:
            qc_cfU_heis_R.x(L)

        qc.append(qc_cfU_heis_R.to_gate(), [i for i in range(L+1)])
        qc.rx(-2*phis[i], L)
    return qc


def cf_heis_U(H1, H2, dagger, c1, c2, nsteps):
    L = int(np.log(len(H1)) / np.log(2))
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    I = np.identity(2)
    
    X_I = X
    for j in range(L):
        X_I = np.kron(X_I, I)
        
    K1 = np.identity(1)
    for j in range(L):
        if j % 2 == 0:
            K1 = np.kron(K1, Z)
        else:
            K1 = np.kron(K1, I)

    K2 = np.identity(1)
    for j in range(L):
        if j % 2 == 0:
            K2 = np.kron(K2, X)
        else:
            K2 = np.kron(K2, I)
    
    # Ks is controlled by the ancilla qubit, applied when qubit is 0.
    cK1  = (1+0j) * np.identity(2*len(H1))
    offset = len(H1)
    for i in range(offset):
        for j in range(offset):
            cK1[i][j] = K1[i][j]
    cK2  = (1+0j) * np.identity(2*len(H2))
    for i in range(offset):
        for j in range(offset):
            cK2[i][j] = K2[i][j]
    
    t = c1/2
    dt = t/nsteps
    cU1 = (1+0j) * np.identity(2*len(H1))
    cU2 = (1+0j) * np.identity(2*len(H2))    
    U1 = expm(-1j*dt*H1)
    U2 = expm(-1j*dt*H2)
    U_ext1 = np.kron(I, U1)
    U_ext2 = np.kron(I, U2)
    cU = np.identity(2*len(H1))
    cU1 = cK1 @ U_ext1 @ cK1
    cU2 = cK2 @ U_ext2 @ cK2

    for i in range(nsteps):
        cU = cU1 @ cU2 @ cU
    
    # Shift the global phase, controlled by the ancilla.
    cC2 = (1+0j) * np.identity(2*len(H1))
    C2 = expm(-1j*c2*np.identity(offset))
    for i in range(offset):
        for j in range(offset):
            cC2[offset+i][offset+j] = C2[i][j]
    cU = cC2 @ cU
    
    if dagger:
        cU = X_I @ cU @ X_I
    return cU


def cf_heis_U_singleK(H, dagger, c1, c2):
    L = int(np.log(len(H)) / np.log(2))
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    I = np.identity(2)
    
    X_I = X
    for j in range(L):
        X_I = np.kron(X_I, I)
        
    # This K does not work!
    K_s = np.kron(X@Y, X@Z)
    K = np.identity(1)
    for j in range(L//2):
        K = np.kron(K, K_s)
    K_dag = K.conj().T
    
    # K is controlled by the ancilla qubit, applied when qubit is 0.
    cK  = (1+0j) * np.identity(2*len(H))
    offset = len(H)
    for i in range(offset):
        for j in range(offset):
            cK[i][j] = K[i][j]

    # K_dag is controlled by the ancilla qubit, applied when qubit is 0.
    cK_dag  = (1+0j) * np.identity(2*len(H))
    for i in range(offset):
        for j in range(offset):
            cK_dag[i][j] = K_dag[i][j]
    
    cU = (1+0j) * np.identity(2*len(H))
    U = expm(-1j*c1/2*H)
    U_ext = np.kron(I, U)
    cU = cK_dag @ U_ext @ cK
    
    # Shift the global phase, controlled by the ancilla.
    cC2 = (1+0j) * np.identity(2*len(H))
    C2 = expm(-1j*c2*np.identity(offset))
    for i in range(offset):
        for j in range(offset):
            cC2[offset+i][offset+j] = C2[i][j]
    cU = cC2 @ cU
    
    if dagger:
        cU = X_I @ cU @ X_I
    return cU


def QETU_heis_cf(H1, H2, phis, c1=1, c2=0, nsteps=1):
    Q = S(phis[len(phis)-1], len(H1))
    for i in range(1, len(phis)):
        Q = Q @ cf_heis_U_singleK(H1, i % 2 == 0, c1, c2) @ S(phis[len(phis)-1-i], len(H1)) 
    return Q


class NoiseAmplificationTranslator(GatesInBasis):
    """A transpiler pass to insert identity by r in place of CX, CY and CZ gates"""
    def __init__(self, r, *args, **kwargs):
         self.r = r["r"]
         self.r_arr = r["r_arr"] 
         super().__init__(*args, **kwargs)


    def run(self, dag):
        """Run the pass."""
        Nc = 0
        for node in dag.op_nodes():
            if node.op.name in ["cx", "cy", "cz"]:
                Nc += 1        
        print(Nc)
        
        j = 0
        # iterate over all operations
        for node in dag.op_nodes():
            if node.op.name in ["cx", "cy", "cz"]:
                r = self.r_arr[j] if len(self.r_arr) else self.r
                # calculate the replacement
                replacement = QuantumCircuit(2)

                if node.op.name == "cx":
                    for i in range(r):
                        replacement.cx(0, 1)
                elif node.op.name == "cy":
                    for i in range(r):
                        replacement.cy(0, 1)
                else:
                    for i in range(r):
                        replacement.cz(0, 1)
                # replace the node with our new decomposition
                dag.substitute_node_with_dag(node, circuit_to_dag(replacement))
                j += 1
        return dag





    

    
