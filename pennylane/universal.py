import json
import pennylane as qml
import pennylane.numpy as np

np.random.seed(1967)

def get_matrix(params):
    """
    Args:
        - params (array): The four parameters of the model.

    Returns:
        - (matrix): The associated matrix to these parameters.
    """


    alpha, beta, gamma, phi = params
    # Put your code here #
    unitary=np.exp(1j*phi)*np.linalg.multi_dot([qml.RZ(gamma,0),qml.RX(beta,0),qml.RX(alpha,0)])
    # Return the matrix
    return unitary
def error(U, params):
    """
    This function determines the similarity between your generated matrix and
    the target unitary.
    Args:
        - U (np.array): Goal matrix that we want to approach.
        - params (array): The four parameters of the model.

    Returns:
        - (float): Error associated with the quality of the solution.
    """

    matrix = get_matrix(params)


    # Put your code here #
    return abs(U-matrix)
    # Return the error


def train_parameters(U):
    epochs = 1000
    lr = 0.01

    grad = qml.grad(error, argnum=1)
    params = np.random.rand(4) * np.pi

    for epoch in range(epochs):
        params -= lr * grad(U, params)

    return params


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    matrix = json.loads(test_case_input)
    params = [float(p) for p in train_parameters(matrix)]
    return json.dumps(params)


def check(solution_output: str, expected_output: str) -> None:
    matrix1 = get_matrix(json.loads(solution_output))
    matrix2 = json.loads(expected_output)
    assert not np.allclose(get_matrix(np.random.rand(4)), get_matrix(np.random.rand(4)))
    assert np.allclose(matrix1, matrix2, atol=0.2)


# These are the public test cases
test_cases = [
    ('[[ 0.70710678,  0.70710678], [ 0.70710678, -0.70710678]]', '[[ 0.70710678,  0.70710678], [ 0.70710678, -0.70710678]]'),
    ('[[ 1,  0], [ 0, -1]]', '[[ 1,  0], [ 0, -1]]')
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
