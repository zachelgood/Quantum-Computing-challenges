import json
from numpy import shape
import pennylane as qml
from pennylane.math import requires_grad
import pennylane.numpy as np
def create_Hamiltonian(h):
    """
    Function in charge of generating the Hamiltonian of the statement.

    Args:
        h (float): magnetic field strength

    Returns:
        (qml.Hamiltonian): Hamiltonian of the statement associated to h
    """
    # Put your code here #
    N=4 #number of sites
    obs=[qml.PauliZ(i)@qml.PauliZ((i+1)%N) for i in range(N)]
    coeff=[-1]*N
    obs+=[qml.PauliX(i) for i in range(N)]
    coeff += [-h]*N
    return qml.Hamiltonian(coeff,obs)
dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def model(params, H):
    """
    To implement VQE you need an ansatz for the candidate ground state!
    Define here the VQE ansatz in terms of some parameters (params) that
    create the candidate ground state. These parameters will
    be optimized later.

    Args:
        params (numpy.array): parameters to be used in the variational circuit
        H (qml.Hamiltonian): Hamiltonian used to calculate the expected value

    Returns:
        (float): Expected value with respect to the Hamiltonian H
    """
    gates=[qml.RX,qml.RY,qml.RZ]
    for i in range(3):
        qml.BasicEntanglerLayers(weights=params[i], wires=range(4), rotation=gates[i])


    return qml.expval(H)

    # Put your code here #
def train(h):
    """
    In this function you must design a subroutine that returns the
    parameters that best approximate the ground state.

    Args:
        h (float): magnetic field strength

    Returns:
        (numpy.array): parameters that best approximate the ground state.
    """

    # Put your code here #
    op=qml.GradientDescentOptimizer(stepsize=0.01)
    maxstep=600
    Ham=create_Hamiltonian(h)
    params = np.random.rand(3,3,4, requires_grad=True)

    # optimising loop
    for _ in range(maxstep): # max iterations
        params = op.step(lambda x: model(x, Ham), params)
    return params
    # for i in range(maxstep):
    #     params, prev_cost=op.step_and_cost(model,params,Ham)
    #     n_cost=model(params,Ham)
    #     if np.abs(n_cost-prev_cost)<=converg:
    #         break
    # return params
# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    params = train(ins)
    return str(model(params, create_Hamiltonian(ins)))


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-1
    ), "The expected value is not correct."

# These are the public test cases
test_cases = [
    ('1.0', '-5.226251859505506'),
    ('2.3', '-9.66382463698038')
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
