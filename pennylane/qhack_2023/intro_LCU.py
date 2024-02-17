import json
import pennylane as qml
import numpy as np
def W(alpha, beta):
    """ this function returns the matrix w in terms of
    the coefficients alpha and beta

    args:
        -alpha (float): the prefactor alpha of u in the linear combination, as in the
        challenge statement.
        - beta (float): the prefactor beta of v in the linear combination, as in the
        challenge statement.
    returns
        -(numpy.ndarray): a 2x2 matrix representing the operator w,
        (as defined in the challenge statement)
    """
    return 1/np.sqrt(alpha+beta)*np.array([[np.sqrt(alpha),-np.sqrt(beta)],[np.sqrt(beta),np.sqrt(alpha)]])
dev = qml.device('default.qubit', wires = 2)

@qml.qnode(dev)
def linear_combination(U, V,  alpha, beta):
    """This circuit implements the circuit that probabilistically calculates the linear combination
    of the unitaries.

    Args:
        - U (list(list(float))): A 2x2 matrix representing the single-qubit unitary operator U.
        - V (list(list(float))): A 2x2 matrix representing the single-qubit unitary operator U.
        - alpha (float): The prefactor alpha of U in the linear combination, as above.
        - beta (float): The prefactor beta of V in the linear combination, as above.

    Returns:
        -(numpy.tensor): Probabilities of measuring the computational
        basis states on the auxiliary wire.
    """
    qml.QubitUnitary(W(alpha,beta),0)
    qml.ControlledQubitUnitary(U, control_wires=[0], wires=2, control_values=[1])

    qml.ControlledQubitUnitary(V, control_wires=[0], wires=2, control_values=[0])
    qml.QubitUnitary(np.transpose(W(alpha,beta)),0)
    return qml.probs(0)
    # These functions are responsible for testing the solution.

def run(test_case_input: str) -> str:
    dev = qml.device('default.qubit', wires = 2)
    ins = json.loads(test_case_input)
    output = linear_combination(*ins)[0].numpy()

    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, atol=1e-3
    ), "Your circuit doesn't look quite right "

# These are the public test cases
test_cases = [
    ('[[[ 0.70710678,  0.70710678], [ 0.70710678, -0.70710678]],[[1, 0], [0, -1]], 1, 3]', '0.8901650422902458')
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
