import argparse
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F

from data import load_data


class SimpleConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(SimpleConvolutionalNetwork, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 4, kernel_size=1, stride=1)  # Output with 4 channels

    def forward(self, img):
        x = self.conv1(img)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        return x


def train(train_dataloader, model, optimizer, loss_fn, device):

    for batch, (input, target) in enumerate(train_dataloader):
        input = input.to(device)
        target = target.to(device)

        output = model(input)

        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(validation_dataloader, model, loss_fn, device):
    validation_loss = 0

    for batch, (input, target) in enumerate(validation_dataloader):
        input = input.to(device)
        target = target.to(device)

        output = model(input)

        loss = loss_fn(output, target)
        validation_loss += loss.item()

    print("validation loss: ", validation_loss / len(validation_dataloader))
    return validation_loss / len(validation_dataloader)


def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return "%dh %02dm %02ds" % (hours, minutes, seconds)
    elif minutes > 0:
        return "%dm %02ds" % (minutes, seconds)
    else:
        return "%ds" % seconds


def main(args):
    start_time = time.time()

    train_dataloader, validation_dataloader = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleConvolutionalNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Training
    epochs = args.epochs
    best_test_loss = float('inf')
    test_loss_list = []

    for epoch in range(epochs):
        train(train_dataloader, model, optimizer, loss_fn, device)
        test_loss = validate(validation_dataloader, model, loss_fn, device)
        test_loss_list.append(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'simple_model.pth')

        print(f"Epoch: {epoch}, Test loss: {test_loss:>8f}")

    # Plot results
    x_axis = np.linspace(0, len(test_loss_list), len(test_loss_list))
    plt.plot(x_axis, test_loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.show()

    print(
        f"Best test loss: {best_test_loss:>8f}, Elapsed time: {format_time(time.time() - start_time)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kaggle competition.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--train_loss', type=bool, default=True, help='Print train loss or not')
    _args = parser.parse_args()

    main(_args)
