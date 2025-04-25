import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from trainer import Trainer
from perceptron import Perceptron
from cnnLayer import CNNLayer

if __name__ == "__main__":
    # set random seed for reproducibility
    torch.manual_seed(42)

    # check for gpu availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    # define transformations for training set with augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # define transformations for test set without augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # load mnist dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )

    # create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64,
                             shuffle=False, pin_memory=True)

    # create FNN model
    # input_size = 28 * 28  # MNIST images are 28x28 pixels
    # hidden_size = [128]  # one hidden layer
    # output_size = 10  # 10 classes (digits 0-9)
    # model = Perceptron(
    #     input_size=input_size,
    #     hidden_size=hidden_size,
    #     output_size=output_size,
    #     softmax_dim=-1
    # )

    # define regularization hyperparameters
    dropout_rate = 0.5
    weight_decay = 1e-4

    # create CNN model
    input_channels = 1
    num_classes = 10
    model = CNNLayer(input_channels, num_classes, dropout_rate, weight_decay)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

    # create a learning rate scheduler for overfitting
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10,
        eta_min=1e-6
    )

    # create our trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,  # using test set as validation set
        device=device,
        scheduler=scheduler,
        checkpoint_dir='./checkpoints',
        experiment_name="MNIST_CNN_Model"
    )

    # train the model
    history = trainer.train(
        num_epochs=10,
        log_to_mlflow=True,
        early_stopping_patience=5  # enable early stopping for overfitting
    )

    # uncomment to resume training from a checkpoint
    # don't forget to change the checkpoint path
    # checkpoint_path = './checkpoints/checkpoint_epoch_2.pt'
    # start_epoch = trainer.load_checkpoint(checkpoint_path)
    # trainer.train(num_epochs=10, start_epoch=start_epoch)

    # uncomment to use the best model for inference
    # trainer.load_checkpoint('./checkpoints/best_model.pt')
