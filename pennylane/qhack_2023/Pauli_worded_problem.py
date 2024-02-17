#!/usr/bin/env python3
import json
import pennylane as qml
import numpy as np
from pennylane.devices import null_qubit
import scipy
def abs_dist(rho, sigma):
    """A function to compute the absolute value |rho - sigma|."""
    polar = scipy.linalg.polar(rho - sigma)
    return polar[1]

def word_dist(word):
    """A function which counts the non-identity operators in a Pauli word"""
    return sum(word[i] != "I" for i in range(len(word)))


# Produce the Pauli density for a given Pauli word and apply noise

def noisy_Pauli_density(word, lmbda):
    """
       A subcircuit which prepares a density matrix (I + P)/2**n for a given Pauli
       word P, and applies depolarizing noise to each qubit. Nothing is returned.

    Args:
            word (str): A Pauli word represented as a string with characters I,  X, Y and Z.
            lmbda (float): The probability of replacing a qubit with something random.
    """

    # Put your code here #
    n_qubit=len(word)
    nwire=range(n_qubit)
    wire_map={i:i for i in range(n_qubit)}
    pword=qml.pauli.string_to_pauli_word(word,wire_map)
    psent=qml.pauli.pauli_word_to_matrix(pword, wire_map)
    denmat=1/2**(n_qubit)*(np.eye(2**n_qubit)+psent)

    qml.QubitDensityMatrix(denmat,wires=nwire)

    for wire in nwire:
        qml.DepolarizingChannel(lmbda,wires=wire)
# Compute the trace distance from a noisy Pauli density to the maximally mixed density

def maxmix_trace_dist(word, lmbda):
    """
       A function compute the trace distance between a noisy density matrix, specified
       by a Pauli word, and the maximally mixed matrix.

    Args:
            word (str): A Pauli word represented as a string with characters I, X, Y and Z.
            lmbda (float): The probability of replacing a qubit with something random.

    Returns:
            float: The trace distance between two matrices encoding Pauli words.
    """

    # Put your code here #
    n_qubit = len(word)
    dev=qml.device('default.mixed', wires=n_qubit)
    @qml.qnode(dev)
    def circ(word,lmbda):
        noisy_Pauli_density(word,lmbda)
        return qml.density_matrix(wires=range(n_qubit))
    pauli=circ(word,lmbda)
    ident=1/2**n_qubit*np.eye(2**n_qubit)
    tracedist=1/2*np.trace(abs_dist(pauli,ident))
    return tracedist


def bound_verifier(word, lmbda):
    """
       A simple check function which verifies the trace distance from a noisy Pauli density
       to the maximally mixed matrix is bounded by (1 - lambda)^|P|.

    Args:
            word (str): A Pauli word represented as a string with characters I, X, Y and Z.
            lmbda (float): The probability of replacing a qubit with something random.

    Returns:
            float: The difference between (1 - lambda)^|P| and T(rho_P(lambda), rho_0).
    """

    # Put your code here #
    tdis=maxmix_trace_dist(word,lmbda)
    bound=(1-lmbda)**(word_dist(word))
    return bound-tdis

# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:

    word, lmbda = json.loads(test_case_input)
    output = np.real(bound_verifier(word, lmbda))

    return str(output)


def check(solution_output: str, expected_output: str) -> None:

    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-4
    ), "Your trace distance isn't quite right!"

# These are the public test cases
test_cases = [
    ('["XXI", 0.7]', '0.0877777777777777'),
    ('["XXIZ", 0.1]', '0.4035185185185055'),
    ('["YIZ", 0.3]', '0.30999999999999284'),
    ('["ZZZZZZZXXX", 0.1]', '0.22914458207245006')
]
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
