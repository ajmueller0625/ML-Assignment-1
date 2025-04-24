import numpy as np


class Layer:
    def __init__(self, neurons_size: int, inputs_size: int, activation: str = 'sigmoid', leaky_slope: float = 0.01, softmax_dim: int = 0):
        '''
        Initialize the layer with random weights and biases

        Parameters:
            neurons_size: int
                The number of neurons in the layer
            inputs_size: int
                The number of inputs to the layer
            activation: str
                The activation function to use
            leaky_slope: float
                The slope of the leaky relu activation function
            softmax_dim: int
                The dimension of the softmax activation function
        '''

        # initialize weights and biases with random values
        # use the ha initialization method to initialize the weights if the activation function is relu or leaky relu else use the xavier initialization method
        if activation == 'relu' or activation == 'leaky_relu':
            self._weights: list[float] = np.random.randn(
                neurons_size, inputs_size) * np.sqrt(2 / inputs_size)
        else:
            self._weights: list[float] = np.random.randn(
                neurons_size, inputs_size) * np.sqrt(1 / inputs_size)

        self._biases: list[float] = np.zeros(neurons_size)
        self._activation: str = activation
        self._leaky_slope: float = leaky_slope
        self._softmax_dim: int = softmax_dim

        self._activation_function: dict[str, callable] = {
            'sigmoid': lambda sum: 1.0 / (1.0 + np.exp(-sum)),
            'tanh': lambda sum: np.tanh(sum),
            'relu': lambda sum: np.maximum(0, sum),
            'leaky_relu': lambda sum: np.maximum(self._leaky_slope * sum, sum),
            'softmax': lambda sum: np.exp(sum) / np.sum(np.exp(sum), axis=self._softmax_dim, keepdims=True)
        }

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        '''
        Forward pass

        Parameters:
            inputs: np.ndarray
                The inputs to the layer
        Returns:
            np.ndarray
                The outputs of the layer
        '''
        # calculate the sum of the inputs multiplied by the weights and add the biases
        sum: np.ndarray = np.dot(self._weights, inputs) + self._biases

        if self._activation in self._activation_function:
            return self._activation_function[self._activation](sum)
        else:
            raise ValueError(
                f"Activation function {self._activation} not found")


# test the layer
if __name__ == "__main__":

    test_inputs: np.ndarray = np.array([2.0, -3.0, 5.0])

    # test the layer with different activation functions
    for activation in ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'softmax']:
        layer: Layer = Layer(5, 3, activation=activation)
        print(f'{activation.capitalize()} activation: {layer.forward(test_inputs)}')
    print('')

    # test the layer with different leaky slopes
    for leaky_slope in [0.01, 0.1, 1.0]:
        layer: Layer = Layer(5, 3, activation='leaky_relu',
                             leaky_slope=leaky_slope)
        print(
            f'Leaky ReLU with slope {leaky_slope}: {layer.forward(test_inputs)}')
    print('')

    # test the layer with a batch of inputs
    batch_inputs: np.ndarray = np.random.randn(4, 3)
    layer: Layer = Layer(5, 3, activation='softmax', softmax_dim=0)
    for i, sample in enumerate(batch_inputs):
        print(f'Sample {i+1}: {sample}')
        print(f'Output {i+1}: {layer.forward(sample)}')
