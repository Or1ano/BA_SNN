import networkx as nx
import torch
import numpy as np


def initialize_ba_network(layer_sizes, m0, m):
    G = nx.barabasi_albert_graph(layer_sizes[0], m0, m)
    adjacency_matrix = nx.adjacency_matrix(G).toarray()
    input_size, output_size = layer_sizes[0], layer_sizes[1]
    ba_weights = np.zeros((input_size, output_size))

    for i in range(input_size):
        connected_nodes = np.nonzero(adjacency_matrix[i])[0]
        for j in range(output_size):
            ba_weights[i, j] = 1 if j in connected_nodes else 0

    noParameters = np.count_nonzero(ba_weights == 1)

    return noParameters, ba_weights


def find_first_pos(weight_tensor, value):
    diff = torch.abs(weight_tensor - value)
    idx = torch.argmin(diff)
    return idx.item()


def find_last_pos(tensor, value):
    diff = torch.abs(tensor - value)
    reversed_diff = torch.flip(diff, [0])  # 反转张量
    idx = torch.argmin(reversed_diff)
    last_pos = tensor.shape[0] - idx.item()
    return last_pos


def rewire_mask(weights, no_weights, zeta):
    # Rewire weight matrix

    # Flatten and sort the weights array
    values = torch.flatten(weights)
    sorted_values, _ = torch.sort(values)

    # Calculate the threshold values for removing weights
    first_zero_pos = find_first_pos(values, 0)
    last_zero_pos = find_last_pos(values, 0)
    largest_negative = sorted_values[int((1 - zeta) * first_zero_pos)]
    smallest_positive = sorted_values[
        int(min(values.shape[0] - 1, last_zero_pos + zeta * (values.shape[0] - last_zero_pos)))]

    # Rewire the weights
    rewired_weights = weights.clone()
    rewired_weights[rewired_weights > smallest_positive] = 1
    rewired_weights[rewired_weights < largest_negative] = 1
    rewired_weights[rewired_weights != 1] = 0
    weight_mask_core = rewired_weights.clone()

    # Add random weights
    nr_add = 0
    no_rewires = no_weights - torch.sum(rewired_weights)
    while nr_add < no_rewires:
        i = np.random.randint(0, rewired_weights.shape[0])
        j = np.random.randint(0, rewired_weights.shape[1])
        if rewired_weights[i, j] == 0:
            rewired_weights[i, j] = 1
            nr_add += 1

    return rewired_weights, weight_mask_core





