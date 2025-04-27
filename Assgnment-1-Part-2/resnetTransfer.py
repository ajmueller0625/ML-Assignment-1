import torch
import torch.nn as nn
import torchvision.models as models

class ResNetTransfer(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        '''
            Initialize transfer learning model using ResNet50
            Parameters:
                input_channels: Number of channels in the input images
                num_classes: Number of output classes
        '''
        super(ResNetTransfer, self).__init__()
        
        # load pre-trained ResNet50 model
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # modify the first layer to accept grayscale images
        original_layer = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=original_layer.out_channels, 
            kernel_size=original_layer.kernel_size, 
            stride=original_layer.stride, 
            padding=original_layer.padding, 
            bias=original_layer.bias is not None
        )
        
        # average weight across channels for grayscale images
        if input_channels == 1:
            with torch.no_grad():
                self.resnet.conv1.weight.data = nn.Parameter(
                    original_layer.weight.data.mean(dim=1, keepdim=True)
                )

        # replace the final classification layer
        self.resnet.fc = nn.Linear(
            in_features=self.resnet.fc.in_features, 
            out_features=num_classes
        )

    def forward(self, input):
        '''
            Forward pass of the model
            Parameters:
                input: Input images
            Returns:
                output: Output of the model
            '''
        return self.resnet(input)
        
        
        
        
        