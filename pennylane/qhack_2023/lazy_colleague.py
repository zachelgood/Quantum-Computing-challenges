#!/usr/bin/env python3
import json
import pennylane as qml
import pennylane.numpy as np
dev = qml.device("default.qubit", wires=["e1", "e2", "e3", "e4", "result"], shots=1)

wires = ["e1", "e2", "e3", "e4", "result"]

@qml.qnode(dev)
def circuit(project_execution):
    """This is the circuit we will use to detect which is the lazy worker. Remember
    that we will only execute one shot.

    Args:
        project_execution (qml.ops):
            The gate in charge of marking in the last qubit if the project has been finished
            as indicated in the statement.

    Returns:
        (numpy.tensor): Measurement output in the 5 qubits after a shot.
    """

    # Put your code here #
    ## This is Grover search over the reduced 4-dimensional subspace
    n = len(wires) - 1
    for i in range(n): # prep i-th qubit
        if i == 0: # non-controlled RX
            qml.RX(np.arcsin(1/np.sqrt(n-i)) * 2, wires=wires[i])
        else: # multi-controlled RX
            qml.ctrl(qml.RX, control=wires[:i])(
                np.arcsin(1/np.sqrt(n-i)) * 2, wires[i]
            )
        qml.PauliX(wires=wires[i]) # rotate i-th qubit to 1

    # phase flip the solution
    qml.PauliX(wires="result")
    qml.Hadamard(wires="result")
    project_execution(wires=wires)

    # reverse the initialisation
    for i in range(n)[::-1]: # prep i-th qubit
        qml.PauliX(wires=wires[i])
        if i == 0: # non-controlled RX
            qml.RX(-np.arcsin(1/np.sqrt(n-i)) * 2, wires=wires[i])
        else: # multi-controlled RX
            qml.ctrl(qml.RX, control=wires[:i])(
                -np.arcsin(1/np.sqrt(n-i)) * 2, wires[i]
            )

    # phase flip all the initial state |0000>
    for i in wires[:-1]:
        qml.PauliX(wires=i)
    qml.ctrl(qml.PauliZ, control=wires[:-2])(wires=wires[-2])
    for i in wires[:-1]:
        qml.PauliX(wires=i)

    # redo the initialisation
    for i in range(n): # prep i-th qubit
        if i == 0: # non-controlled RX
            qml.RX(np.arcsin(1/np.sqrt(n-i)) * 2, wires=wires[i])
        else: # multi-controlled RX
            qml.ctrl(qml.RX, control=wires[:i])(
                np.arcsin(1/np.sqrt(n-i)) * 2, wires[i]
            )
        qml.PauliX(wires=wires[i])

    return qml.sample(wires=dev.wires[:-1])
    
    # Put your code here #

def process_output(output):
    """This function will take the circuit measurement and process it to determine who is the lazy worker.

    Args:
        output (numpy.tensor): Measurement output in the 5 qubits after a shot.

    Returns:
        (str): This function must return "e1", "e2" "e3" or "e4" - the lazy worker.
    """

    # Put your code here #
    if np.all(output == [0,1,1,1]):
        return "e1"
    elif np.all(output == [1,0,1,1]):
        return "e2"
    elif np.all(output == [1,1,0,1]):
        return "e3"
    else:
        return "e4"

# These functions are responsible for testing the solution.

def run(test_case_input: str) -> str:
    return None

def check(solution_output: str, expected_output: str) -> None:
    samples = 5000

    solutions = []
    output = []

    for s in range(samples):
        lazy = np.random.randint(0, 4)
        no_lazy = list(range(4))
        no_lazy.pop(lazy)

        def project_execution(wires):
            class op(qml.operation.Operator):
                num_wires = 5

                def compute_decomposition(self, wires):
                    raise ValueError("You cant descompose this gate")

                def matrix(self):
                    m = np.zeros([32, 32])
                    for i in range(32):
                        b = [int(j) for j in bin(64 + i)[-5:]]
                        if sum(np.array(b)[no_lazy]) == 3:
                            if b[-1] == 0:
                                m[i, i + 1] = 1
                            else:
                                m[i, i - 1] = 1
                        else:
                            m[i, i] = 1
                    return m

            op(wires=wires)
            return None

        out = circuit(project_execution)
        solutions.append(lazy + 1)
        output.append(int(process_output(out)[-1]))

    assert np.allclose(
        output, solutions, rtol=1e-4
    ), "Your circuit does not give the correct output."

    ops = [op.name for op in circuit.tape.operations]
    assert ops.count("op") == 1, "You have used the oracle more than one time."

# These are the public test cases
test_cases = [
    ('No input', 'No output')
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
