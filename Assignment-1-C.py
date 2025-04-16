import torch
import torch.nn as nn


class Layer:
    def __init__(self, neurons_size: int, inputs_size: int, activation: str = 'sigmoid'):
        '''
        Initialize the layer with random weights and biases

        Parameters:
            neurons_size: int
                The number of neurons in the layer
            inputs_size: int
                The number of inputs to the layer
        '''

        # initialize weights and biases with random values
        # multiply by 0.1 to avoid large weights
        self._weights: torch.Tensor = torch.randn(
            neurons_size, inputs_size) * 0.1
        self._biases: torch.Tensor = torch.zeros(neurons_size)
        self._activation: str = activation

    def _activation_function(self, sum: torch.Tensor, activation: str) -> torch.Tensor:
        '''
        Apply the activation function to the sum

        Parameters:
            sum: torch.Tensor
                The sum of the inputs
        Returns:
            torch.Tensor
                The output of the activation function
        '''
        # use the activation function specified in the constructor
        if activation == 'sigmoid':
            return self._sigmoid(sum)
        elif activation == 'tanh':
            return self._tanh(sum)
        elif activation == 'relu':
            return self._relu(sum)
        elif activation == 'leaky_relu':
            return self._leaky_relu(sum)
        else:
            raise ValueError(f"Activation function {activation} not found")

    def _sigmoid(self, sum: torch.Tensor) -> torch.Tensor:
        '''
        Apply the sigmoid activation function to the sum

        Parameters:
            sum: torch.Tensor
                The sum of the inputs
        Returns:
            torch.Tensor
                The output of the activation function
        '''
        # return based on the sigmoid function where the output is 1 divided by 1 plus e to the power of -z(sum)
        # return 1.0 / (1.0 + torch.exp(-sum))
        return nn.Sigmoid()(sum)

    def _tanh(self, sum: torch.Tensor) -> torch.Tensor:
        '''
        Apply the tanh activation function to the sum

        Parameters:
            sum: torch.Tensor
                The sum of the inputs
        Returns:
            torch.Tensor
                The output of the activation function
        '''
        # return based on the tanh function where the output is the hyperbolic tangent of the sum
        # return torch.tanh(sum)
        return nn.Tanh()(sum)

    def _relu(self, sum: torch.Tensor) -> torch.Tensor:
        '''
        Apply the relu activation function to the sum

        Parameters:
            sum: torch.Tensor
                The sum of the inputs
        Returns:
            torch.Tensor
                The output of the activation function
        '''
        # return based on the relu function where the output is the maximum of 0 and the sum
        # return torch.maximum(torch.zeros_like(sum), sum)
        return nn.ReLU()(sum)

    def _leaky_relu(self, sum: torch.Tensor) -> torch.Tensor:
        '''
        Apply the leaky relu activation function to the sum

        Parameters:
            sum: torch.Tensor
                The sum of the inputs
        Returns:
            torch.Tensor
                The output of the activation function
        '''
        # return based on the leaky relu function where the output is the maximum of 0.01 multiplied by the sum and the sum
        # return torch.maximum(0.01 * sum, sum)
        return nn.LeakyReLU(negative_slope=0.01)(sum)

    def forward(self, inputs: torch.Tensor, activation: str = 'sigmoid') -> torch.Tensor:
        '''
        Forward pass

        Parameters:
            inputs: torch.Tensor
                The inputs to the layer
            activation: str
                The activation function to use
        Returns:
            torch.Tensor
                The outputs of the layer
        '''
        # calculate the sum of the inputs multiplied by the weights and add the biases
        # sum: torch.Tensor = torch.matmul(self._weights, inputs) + self._biases
        sum: torch.Tensor = self._weights @ inputs + self._biases
        return self._activation_function(sum, activation)


# test the layer
if __name__ == "__main__":
    layer: Layer = Layer(5, 3)
    inputs: torch.Tensor = torch.tensor([2.0, -3.0, 5.0])
    print(f'Sigmoid activation: {layer.forward(inputs, 'sigmoid')}')
    print(f'Tanh activation: {layer.forward(inputs, 'tanh')}')
    print(f'ReLU activation: {layer.forward(inputs, 'relu')}')
    print(f'Leaky ReLU activation: {layer.forward(inputs, 'leaky_relu')}')
