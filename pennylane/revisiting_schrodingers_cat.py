#!/usr/bin/env python3

import json
import pennylane as qml
import pennylane.numpy as np

dev = qml.device("default.qubit", wires=["atom", "cat"])


@qml.qnode(dev)
def evolve_atom_cat(unitary, params):
    """
    Circuit that implements the evolution of the atom-cat system under the action of a unitary
    and the change of basis in the atom wire before measuring.

    Args:
        unitary (np.array(complex)): The matrix of a 4x4 unitary operator.
        params (list(float)): A list of three angles corresponding to the parameters
        of the U3 gate

    Returns:
        (np.tensor): The state of the joint atom-cat system after unitary evolution.
    """

    # Put your code here #
    qml.QubitUnitary(unitary, wires=["atom", "cat"])
    qml.U3(*params, wires="atom")
    return qml.state()


def u3_parameters(unitary):
    """
    Find adequate parameters that yield a uniform position on the cat wire
    when the atom wire is measured to be |0>.

    Args:
        unitary (np.array(complex)): The matrix of a 4x4 unitary operator.

    Returns:
        (np.array(float)): The parameters for the U3 change of basis that yield
        a uniform superposition for the cat when the atom is measured in the
        state |0>.
    """

    # Put your code here #

    # Return a set of parameters that satisfy the required condition
    def cost(params, unitary):
        state = np.array(evolve_atom_cat(unitary, params))
        print(f"State: {state}")
        err = np.abs(state[0] - state[1])
        print(f"Cost: {err}")
        return err

    params = np.array([0.1, 0.2, 0.3], requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.01)
    steps = 300
    for i in range(steps):
        params, prev_err = opt.step_and_cost(lambda p: cost(p, unitary), params)
        print(f"Parameters at step {i}: {params}")
        print(f"Cost at step {i}: {prev_err}")
        if prev_err < 1e-6:
            break

    return params


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:

    ins = json.loads(test_case_input)
    output = u3_parameters(ins).tolist()

    if np.isclose(
        evolve_atom_cat(ins, output)[0], evolve_atom_cat(ins, output)[1], atol=5e-2
    ):

        return "Cat state generated"

    return "Cat state not generated"


def check(solution_output: str, expected_output: str) -> None:

    def unitary_circ():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])

    U1 = qml.matrix(unitary_circ, wire_order=[0, 1])()

    assert np.isclose(
        evolve_atom_cat(U1, [1, 1, 1])[0], 0.62054458
    ), "Your evolve_atom_cat circuit does not do what is expected."
    assert (
        solution_output == expected_output
    ), "Your parameters do not generate a Schrodinger cat"


# These are the public test cases
test_cases = [
    (
        "[[ 0.70710678,  0 ,  0.70710678,  0], [0 ,0.70710678, 0, 0.70710678], [ 0,  0.70710678,  0, -0.70710678], [ 0.70710678,  0, -0.70710678,  0]]",
        "Cat state generated",
    ),
    (
        "[[-0.00202114,  0.99211964, -0.05149589, -0.11420469], [-0.13637119, -0.1236727 , -0.30532593, -0.93428263], [0.89775373,  0.00794205, -0.363445  ,  0.24876274], [ 0.41885207, -0.01845563, -0.8786535 ,  0.22845207]]",
        "Cat state generated",
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
