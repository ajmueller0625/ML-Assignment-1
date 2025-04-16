import numpy as np


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
        self._weights: list[float] = np.random.randn(
            neurons_size, inputs_size) * 0.1
        self._biases: list[float] = np.zeros(neurons_size)
        self._activation: str = activation

    def _activation_function(self, sum: np.ndarray, activation: str) -> np.ndarray:
        '''
        Activation function

        Parameters:
            sum: np.ndarray
                The sum of the inputs
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

    def _sigmoid(self, sum: np.ndarray) -> np.ndarray:
        '''
        Sigmoid activation function

        Parameters:
            sum: np.ndarray
                The sum of the inputs
        Returns:
            np.ndarray
                The output of the activation function
        '''
        # return based on the sigmoid function where the output is 1 divided by 1 plus e to the power of -z(sum)
        return 1.0 / (1.0 + np.exp(-sum))

    def _tanh(self, sum: np.ndarray) -> np.ndarray:
        '''
        Tanh activation function

        Parameters:
            sum: np.ndarray
                The sum of the inputs
        Returns:
            np.ndarray
                The output of the activation function
        '''
        # return based on the tanh function where the output is the hyperbolic tangent of the sum
        return np.tanh(sum)

    def _relu(self, sum: np.ndarray) -> np.ndarray:
        '''
        ReLU activation function

        Parameters:
            sum: np.ndarray
                The sum of the inputs
        Returns:
            np.ndarray
                The output of the activation function
        '''
        # return based on the relu function where the output is the maximum of 0 and the sum
        return np.maximum(0, sum)

    def _leaky_relu(self, sum: np.ndarray) -> np.ndarray:
        '''
        Leaky ReLU activation function

        Parameters:
            sum: np.ndarray
                The sum of the inputs
        Returns:
            np.ndarray
                The output of the activation function
        '''
        # return based on the leaky relu function where the output is the maximum of 0.01 multiplied by the sum and the sum
        return np.maximum(0.01 * sum, sum)

    def forward(self, inputs: np.ndarray, activation: str = 'sigmoid') -> np.ndarray:
        '''
        Forward pass

        Parameters:
            inputs: np.ndarray
                The inputs to the layer
            activation: str
                The activation function to use
        Returns:
            np.ndarray
                The outputs of the layer
        '''
        # calculate the sum of the inputs multiplied by the weights and add the biases
        sum: np.ndarray = np.dot(self._weights, inputs) + self._biases
        return self._activation_function(sum, activation)


# test the layer
if __name__ == "__main__":

    layer: Layer = Layer(5, 3)
    inputs: np.ndarray = np.array([2.0, -3.0, 5.0])
    print(f'Sigmoid activation: {layer.forward(inputs, "sigmoid")}')
    print(f'Tanh activation: {layer.forward(inputs, "tanh")}')
    print(f'ReLU activation: {layer.forward(inputs, "relu")}')
    print(f'Leaky ReLU activation: {layer.forward(inputs, "leaky_relu")}')
