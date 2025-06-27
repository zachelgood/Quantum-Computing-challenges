#!/usr/bin/env python3
import json
import pennylane as qml
import pennylane.numpy as np

wires_m = [0, 1, 2]  # qubits needed to encode m
wires_n = [3, 4, 5]  # qubits needed to encode n
wires_aux = [6, 7, 8, 9, 10]  # auxiliary qubits you can use

# Put your code here #


# Create all the helper functions you need here
def wire_control(control, target):
    qml.CNOT([control, target])
    qml.MultiControlledX(
        [
            control,
            target,
        ]
    )


def oracle_distance(d):
    """
    Args:
        d (int): the distance with which we will check that the oracle is working properly.

    This function does not return anything, it is a quantum function that applies
    necessary gates that implement the requested oracle.

    """

    # Put your code here
    qml.CNOT(wires=[5, 2])
    qml.MultiControlledX(wires=[2, 5, 7], control_values=[1, 1])
    qml.CNOT(wires=[2, 8])
    qml.MultiControlledX(wires=[2, 8, 7], control_values=[0, 1])
    qml.CNOT(wires=[5, 2])

    qml.CNOT(wires=[4, 1])
    qml.MultiControlledX(wires=[1, 4, 6], control_values=[1, 1])
    qml.CNOT(wires=[1, 7])
    qml.MultiControlledX(wires=[1, 7, 6], control_values=[0, 1])
    qml.CNOT(wires=[4, 1])

    qml.CNOT(wires=[3, 0])
    qml.MultiControlledX(wires=[0, 3, 9], control_values=[1, 1])
    qml.CNOT(wires=[0, 6])
    qml.MultiControlledX(wires=[0, 6, 9], control_values=[0, 1])
    qml.CNOT(wires=[3, 0])

    qml.MultiControlledX(wires=[8, 9, 7], control_values=[1, 1])
    qml.CNOT(wires=[9, 6])
    qml.MultiControlledX(wires=[9, 7, 8, 6], control_values=[1, 0, 0])

    if d >= 4:
        qml.PauliX(wires=6)
        d = d - 4
    if d >= 2:
        qml.PauliX(wires=7)
        d = d - 2
    if d >= 1:
        qml.PauliX(wires=8)

    for qubit in [6, 7, 8]:
        qml.PauliX(wires=qubit)
        # Phase flip
    qml.CCZ(wires=[6, 7, 8])


# These functions are responsible for testing the solution.
wires_m = [0, 1, 2]
wires_n = [3, 4, 5]
wires_aux = [6, 7, 8, 9, 10]

dev = qml.device("default.qubit.legacy", wires=11)


@qml.qnode(dev)
def circuit(m, n, d):
    qml.BasisEmbedding(m, wires=wires_m)
    qml.BasisEmbedding(n, wires=wires_n)
    oracle_distance(d)
    return qml.state()


def run(test_case_input: str) -> str:
    outputs = []
    d = int(json.loads(test_case_input))
    for n in range(8):
        for m in range(8):
            outputs.append(sum(circuit(n, m, d)).real)
    output_list = [round(elem.numpy()) for elem in outputs]
    return str(output_list)


def check(solution_output: str, expected_output: str) -> None:
    i = 0
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)

    assert np.allclose(solution_output, expected_output)

    circuit(np.random.randint(7), np.random.randint(7), np.random.randint(7))
    tape = circuit.qtape

    names = [op.name for op in tape.operations]

    assert names.count("QubitUnitary") == 0, "Can't use custom-built gates!"


# These are the public test cases
test_cases = [
    (
        "0",
        "[-1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1]",
    ),
    (
        "1",
        "[1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1]",
    ),
    (
        "2",
        "[1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1]",
    ),
    (
        "3",
        "[1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1]",
    ),
    (
        "4",
        "[1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1]",
    ),
    (
        "5",
        "[1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1]",
    ),
    (
        "6",
        "[1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1]",
    ),
    (
        "7",
        "[1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1]",
    ),
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
