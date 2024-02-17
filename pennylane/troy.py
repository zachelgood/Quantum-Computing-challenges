import pennylane as qml
import numpy as np
dev = qml.device("default.qubit", wires=1)

def exact_result_XandZ(alpha, beta, time):
    """Exact circuit for evolving a qubit with H = alpha Z + beta X.

    Args:
        alpha (float): The coefficient of Z in the Hamiltonian.
        beta (float): The coefficient of X in the Hamiltonian.
        time (float): The time we evolve the state for.

    Returns:
        array[complex]: The exact state after evolution.
    """
    root = np.sqrt(alpha**2 + beta**2)
    c_0 = np.cos(root*time) - (alpha/root)*np.sin(root*time)*1.j
    c_1 = -(beta/root)*np.sin(root*time)*1.j
    return np.array([c_0, c_1])

@qml.qnode(dev)
def trotter_XandZ(alpha, beta, time, n):
    """Trotterized circuit for evolving a qubit with H = alpha Z + beta X.

    Args:
        alpha (float): The coefficient of Z in the Hamiltonian.
        beta (float): The coefficient of X in the Hamiltonian.
        time (float): The time we evolve the state for.
        n (int): The number of steps in our Trotterization.

    Returns:
        array[complex]: The state after applying the Trotterized circuit.
    """
    ##################
    # YOUR CODE HERE #
    ##################
    coeffs = [alpha, beta]
    obs = [qml.Z(0), qml.X(0)]
    H = qml.Hamiltonian(coeffs, obs)
    qml.ApproxTimeEvolution(H,time,n)
    return qml.state()
def trotter_error_XandZ(alpha, beta, time, n):
    """Difference between the exact and Trotterized result.

    Args:
        alpha (float): The coefficient of Z in the Hamiltonian.
        beta (float): The coefficient of X in the Hamiltonian.
        time (float): The time we evolve the state for.
        n (int): The number of steps in our Trotterization.

    Returns:
        float: The distance between the exact and Trotterized result.
    """
    diff = np.abs(trotter_XandZ(alpha, beta, time, n) - exact_result_XandZ(alpha, beta, time))
    return np.sqrt(sum(diff*diff))
