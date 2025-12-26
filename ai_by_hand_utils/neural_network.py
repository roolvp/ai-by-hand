# Layer: weights + biases + Activation Function
# Concatenation or structure , list of layers


from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class Layer:
    weights: np.matrix
    bias: np.matrix
    activation_f : Callable

class NeuralNetwork():

    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward_propagation(self, input):
        '''
        Processes the multiplications of Input and Layers in the neural network
        input : X

        '''
        print("Input:")
        print(input)
        results = input
        for layer in self.layers:
            print("Muiltiplying with weights:")
            print(layer.weights)
            print(f"Weights size: {layer.weights.shape}  ,  input size: {results.T.size}")

            results = layer.activation_f(np.dot(layer.weights, results.T) + layer.bias.T) 
            
            # Transponsing it to work with the patter so far
            results = results.T
            print(results)
        return results
    