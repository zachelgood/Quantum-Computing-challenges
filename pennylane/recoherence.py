#!/usr/bin/env python3

import json
from numpy import isclose
import pennylane as qml
import pennylane.numpy as np

dev = qml.device("default.qubit", wires=5)


@qml.qnode(dev)
def evolve_state(coeffs, time):
    """
    Args:
        coeffs (list(float)): A list of the coupling constants g_1, g_2, g_3, and g_4
        time (float): The evolution time of th system under the given Hamiltonian

    Returns:
        (numpy.tensor): The density matrix for the evolved state of the central spin.
    """

    # We build the Hamiltonian for you

    operators = [
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.PauliZ(0) @ qml.PauliZ(2),
        qml.PauliZ(0) @ qml.PauliZ(3),
        qml.PauliZ(0) @ qml.PauliZ(4),
    ]
    hamiltonian = qml.dot(coeffs, operators)
    # add alpha coeffs
    alpha = [0.5 * np.pi, 0.4, 1.2, 1.8, 1.6]
    # evolve Hamiltonian
    qml.RY(alpha[0], wires=0)
    qml.RY(alpha[1], wires=1)
    qml.RY(alpha[2], wires=2)
    qml.RY(alpha[3], wires=3)
    qml.RY(alpha[4], wires=4)
    hamiltonian2 = qml.Hamiltonian(coeffs, operators)
    qml.CommutingEvolution(hamiltonian2, time)
    return qml.density_matrix(0)
    # Return the required density matrix.


def purity(rho):
    """
    Args:
        rho (array(array(complex))): An array-like object representing a density matrix

    Returns:
        (float): The purity of the density matrix rho

    """

    # Put your code here
    return qml.math.purity(rho, [0])
    # Return the purity


def recoherence_time(coeffs):
    """
    Args:
        coeffs (list(float)): A list of the coupling constants g_1, g_2, g_3, and g_4.

    Returns:
        (float): The recoherence time of the central spin.

    """

    # Return the recoherence time
    maxiter = 5000
    tstep = 0.1
    iteration = 0
    while iteration < maxiter:
        rho = evolve_state(coeffs, tstep)
        if np.isclose(purity(rho), 1, rtol=1e-2):
            break
        iteration += 1
        tstep += 0.02
    return tstep


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    params = json.loads(test_case_input)
    output = recoherence_time(params)

    return str(output)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)

    assert np.isclose(solution_output, expected_output, rtol=5e-2)


# These are the public test cases
test_cases = [("[5,5,5,5]", "0.314"), ("[1.1,1.3,1,2.3]", "15.71")]
# This will run the public test cases locally
for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")
