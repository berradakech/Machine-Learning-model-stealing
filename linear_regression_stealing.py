import numpy as np
from phe import paillier
from typing import Callable, List

N_PARAMS = 10

def steal_weight_by_weight(n_params: int, query_function: Callable[[List[float]], float]) -> np.ndarray:
    keys = paillier.generate_paillier_keypair()
    bias_input = np.zeros(n_params)
    bias = query_function(bias_input, keys)
    weights = []
    for i in range(n_params):
        input_vector = np.zeros(n_params)
        input_vector[i] = 1
        response = query_function(input_vector, keys) - bias
        weights.append(response)
    return bias, weights

def steal_linear_equation(n_params: int, query_function: Callable[[List[float]], float]) -> np.ndarray:
    keys = paillier.generate_paillier_keypair()
    input_matrix = np.zeros((n_params, n_params))
    for i in range(n_params):
        input_matrix[i, i] = 1
    bias_input = np.zeros(n_params)
    bias = query_function(bias_input, keys)
    encrypted_matrix = []
    for i in range(n_params):
        input_vector = input_matrix[:, i]
        response = query_function(input_vector, keys) - bias
        encrypted_matrix.append(response)
    decrypted_matrix = [keys.decrypt(vec) for vec in encrypted_matrix]
    weights = np.linalg.solve(input_matrix, decrypted_matrix)
    return bias, weights

def main():
    # Stealing the model using the Weight-by-weight method
    print("Stealing the model using the Weight-by-weight method...")
    bias, weights = steal_weight_by_weight(N_PARAMS, query_pred)

    test_input = np.random.rand(N_PARAMS)

    true_prediction = query_pred(test_input)

    clone_prediction = test_input @ weights + bias

    print(f'True prediction = {true_prediction}')
    print(f'Clone prediction = {clone_prediction}')
    print(f'Difference = {true_prediction - clone_prediction}')

    assert 2**(-16) > abs(true_prediction - clone_prediction)

    print('Successfully stolen the model using the Weight-by-weight method')
    print(f'Weights:')
    [print(weight) for weight in weights]
    print(f'Bias: {bias}')

    # Stealing the model using the Linear equation method
    print("Stealing the model using the Linear equation method...")
    bias, weights = steal_linear_equation(N_PARAMS, query_pred)

    test_input = np.random.rand(N_PARAMS)

    true_prediction = query_pred(test_input)

    clone_prediction = test_input @ weights + bias

    print(f'True prediction = {true_prediction}')
    print(f'Clone prediction = {clone_prediction}')
    print(f'Difference = {true_prediction - clone_prediction}')

    assert 2**(-16) > abs(true_prediction - clone_prediction)

    print('Successfully stolen the model using the Linear equation method')
    print(f'Weights:')
    [print(weight) for weight in weights]
    print(f'Bias: {bias}')

if __name__ == '__main__':
    main()
