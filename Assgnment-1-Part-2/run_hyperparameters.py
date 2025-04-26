import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from trainer import Trainer
from perceptron import Perceptron
from cnnLayer import CNNLayer
import time


def train_with_hyperparams(config, config_id):
    # set random seed for reproducibility
    torch.manual_seed(42)

    # check for gpu availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    print(f'Training with configuration {config_id}\n')

    # define transformations for training set with augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(
            0.1, 0.1), scale=(0.9, 1.1), shear=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # define transformations for test set without augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load mnist dataset
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
        train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    # create cnn model
    model = CNNLayer(
        input_channels=config['input_channels'],
        num_classes=config['num_classes'],
        dropout_rate=config['dropout_rate'],
        weight_decay=config['weight_decay']
    )

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=1e-6
    )

    # create checkpoint directory for this configuration
    checkpoint_dir = f'./checkpoints/config_{config_id}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # create our trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
        experiment_name=f"MNIST_CNN_Config_{config_id}"
    )

    # train the model
    start_time = time.time()
    history = trainer.train(
        num_epochs=config['num_epochs'],
        log_to_mlflow=True,
        early_stopping_patience=config['early_stopping_patience']
    )
    training_time = time.time() - start_time

    # get final test performance
    test_loss, test_accuracy = trainer.test()

    return {
        'config_id': config_id,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'training_time': training_time,
        'epochs_completed': trainer.total_epochs_completed,
        **config  # include all hyperparameters
    }


if __name__ == "__main__":
    # list of hyperparameter configurations to try
    hyperparameter_configs = [
        # configuration 1: Baseline
        {
            'input_channels': 1,
            'num_classes': 10,
            'dropout_rate': 0.5,
            'weight_decay': 1e-4,
            'learning_rate': 0.001,
            'batch_size': 64,
            'num_epochs': 10,
            'early_stopping_patience': 5
        },

        # configuration 2: higher dropout rate
        {
            'input_channels': 1,
            'num_classes': 10,
            'dropout_rate': 0.7,
            'weight_decay': 1e-4,
            'learning_rate': 0.001,
            'batch_size': 64,
            'num_epochs': 10,
            'early_stopping_patience': 5
        },

        # configuration 3: lower dropout rate
        {
            'input_channels': 1,
            'num_classes': 10,
            'dropout_rate': 0.3,
            'weight_decay': 1e-4,
            'learning_rate': 0.001,
            'batch_size': 64,
            'num_epochs': 10,
            'early_stopping_patience': 5
        },

        # configuration 4: higher weight decay
        {
            'input_channels': 1,
            'num_classes': 10,
            'dropout_rate': 0.5,
            'weight_decay': 1e-3,
            'learning_rate': 0.001,
            'batch_size': 64,
            'num_epochs': 10,
            'early_stopping_patience': 5
        },

        # configuration 5: lower weight decay
        {
            'input_channels': 1,
            'num_classes': 10,
            'dropout_rate': 0.5,
            'weight_decay': 1e-5,
            'learning_rate': 0.001,
            'batch_size': 64,
            'num_epochs': 10,
            'early_stopping_patience': 5
        },

        # configuration 6: higher learning rate
        {
            'input_channels': 1,
            'num_classes': 10,
            'dropout_rate': 0.5,
            'weight_decay': 1e-4,
            'learning_rate': 0.01,
            'batch_size': 64,
            'num_epochs': 10,
            'early_stopping_patience': 5
        },

        # configuration 7: lower learning rate
        {
            'input_channels': 1,
            'num_classes': 10,
            'dropout_rate': 0.5,
            'weight_decay': 1e-4,
            'learning_rate': 0.0001,
            'batch_size': 64,
            'num_epochs': 10,
            'early_stopping_patience': 5
        },

        # configuration 8: larger batch size
        {
            'input_channels': 1,
            'num_classes': 10,
            'dropout_rate': 0.5,
            'weight_decay': 1e-4,
            'learning_rate': 0.001,
            'batch_size': 128,
            'num_epochs': 10,
            'early_stopping_patience': 5
        },

        # configuration 9: smaller batch size
        {
            'input_channels': 1,
            'num_classes': 10,
            'dropout_rate': 0.5,
            'weight_decay': 1e-4,
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 10,
            'early_stopping_patience': 5
        },

        # configuration 10: larger number of epochs
        {
            'input_channels': 1,
            'num_classes': 10,
            'dropout_rate': 0.5,
            'weight_decay': 1e-4,
            'learning_rate': 0.001,
            'batch_size': 64,
            'num_epochs': 20,
            'early_stopping_patience': 5
        }
    ]

    # create directory for results
    results_dir = './hyperparameter_results'
    os.makedirs(results_dir, exist_ok=True)

    # train with each configuration and collect results
    results = []
    for i, config in enumerate(hyperparameter_configs):
        result = train_with_hyperparams(config, i+1)
        results.append(result)

    # save results to csv to get the best configuration
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(
        results_dir, 'hyperparameter_results.csv'), index=False)
    results_df_sorted = results_df.sort_values(
        'test_accuracy', ascending=False)

    # find the best configuration
    best_config_id = results_df_sorted.iloc[0]['config_id']
    print(f"\nBest configuration: {best_config_id}")
