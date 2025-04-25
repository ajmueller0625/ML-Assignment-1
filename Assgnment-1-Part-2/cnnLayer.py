import torch
import torch.nn as nn

class CNNLayer(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, dropout_rate=0.5, weight_decay=1e-5):
        '''
        Initialize the CNNLayer class
        
        Parameters:
            input_channels (int): number of channels in the input image
            num_classes (int): number of classes in the output
        '''
        super(CNNLayer, self).__init__()

        # define the convolutional layers
        self.conv_layer = nn.Sequential(
            # create the first convolutional layer
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
            # add a ReLU activation function which is recommended for CNNs
            nn.ReLU(),
            # add a batch normalization layer
            nn.BatchNorm2d(32), 
            # add a max pooling layer which reduces the size of the image by half
            nn.MaxPool2d(kernel_size=2, stride=2),
            # add a dropout layer to prevent overfitting
            nn.Dropout2d(p=dropout_rate/2),

            # create the second convolutional layer
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate/2)
        )
        
        # calculate the size of the flattened layer
        # after the convolutional layer, the image is reduced to 64x7x7
        flattened_size = 64 * 7 * 7

        # define the fully connected layer
        self.fc_layer = nn.Sequential(
            nn.Flatten(), # flatten the image from 3d to 1d
            nn.Linear(flattened_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

        # store the weight decay for optimizer
        self.weight_decay = weight_decay

        # initialize the weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
                    
    def forward(self, inputs):
        '''
        Forward pass of the CNNLayer
        
        Parameters:
            inputs (torch.Tensor): the input image
        '''

        # pass the inputs through the convolutional layer
        outputs = self.conv_layer(inputs)

        # pass the outputs through the fully connected layer
        outputs = self.fc_layer(outputs)

        return outputs

