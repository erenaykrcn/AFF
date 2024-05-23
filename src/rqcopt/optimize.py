import rqcopt as oc
import numpy as np
import scipy
import qib
import h5py


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



def ising1d_dynamics_opt(nlayers: int, t, bootstrap: bool, coeffs_start=[], path="", **kwargs):
    """
    Optimize the quantum gates in a brickwall layout to approximate
    the time evolution governed by an Ising Hamiltonian.
    """
    # side length of lattice
    L = 6
    # Hamiltonian parameters
    J = 1
    g = 1

    # construct Hamiltonian
    latt = qib.lattice.IntegerLattice((L,), pbc=True)
    field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
    H = qib.IsingHamiltonian(field, J, 0, g).as_matrix().todense()

    # reference global unitary
    expiH = scipy.linalg.expm(-1j*H*t)

    # unitaries used as starting point for optimization
    if bootstrap:
        # load optimized unitaries for nlayers - 2 from disk
        with h5py.File(f"../../src/rqcopt/results/ising1d_L{L}_t{t}_dynamics_opt_layers{nlayers-2}.hdf5", "r") as f:
            # parameters must agree
            assert f.attrs["L"] == L
            assert f.attrs["J"] == J
            assert f.attrs["g"] == g
            assert f.attrs["t"] == t
            Vlist_start = f["Vlist"][:]
            assert Vlist_start.shape[0] == nlayers - 2
        # pad identity matrices
        id4 = np.identity(4).reshape((1, 4, 4))
        Vlist_start = np.concatenate((id4, Vlist_start, id4), axis=0)
        assert Vlist_start.shape[0] == nlayers
        perms = [None if i % 2 == 1 else np.roll(range(L), -1) for i in range(len(Vlist_start))]
    else:
        # local Hamiltonian term
        hloc = construct_ising_local_term(J, g)
        assert len(coeffs_start) == nlayers
        Vlist_start = [scipy.linalg.expm(-1j*c*t*hloc) for c in coeffs_start]
        perms = [None if i % 2 == 0 else np.roll(range(L), -1) for i in range(len(Vlist_start))]
    # perform optimization
    Vlist, f_iter, err_iter = oc.optimize_brickwall_circuit(L, expiH, Vlist_start, perms, **kwargs)

    
    # save results to disk
    f_iter = np.array(f_iter)
    err_iter = np.array(err_iter)
    
    if path == "":
        path = f"../../src/rqcopt/results/ising1d_L{L}_t{t}_dynamics_opt_layers{nlayers}.hdf5"

    with h5py.File(path, "w") as f:
        f.create_dataset("Vlist", data=Vlist)
        f.create_dataset("f_iter", data=f_iter)
        f.create_dataset("err_iter", data=err_iter)
        # store parameters
        f.attrs["L"] = L
        f.attrs["J"] = float(J)
        f.attrs["g"] = float(g)
        f.attrs["t"] = float(t)
    return Vlist


