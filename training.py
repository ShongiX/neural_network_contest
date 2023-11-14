import argparse
import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from monai.networks.nets import UNet
from torch import nn

from data_preprocessing import load_data, create_image
from simple_convolution import SimpleConvolutionalNetwork


def train(train_dataloader, model, optimizer, loss_fn, device):

    for batch, (input, target) in enumerate(train_dataloader):
        input = input.to(device)
        target = target.to(device)

        output = model(input)

        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            plt.subplot(1, 2, 1)
            plt.imshow(create_image(target[0, :, :].cpu().detach().numpy()))
            plt.subplot(1, 2, 2)
            output = torch.argmax(output, dim=1)
            plt.imshow(create_image(output[0, :, :].cpu().detach().numpy()))
            plt.show()


def validate(validation_dataloader, model, loss_fn, device):
    validation_loss = 0

    with torch.no_grad():
        for batch, (input, target) in enumerate(validation_dataloader):
            input = input.to(device)
            target = target.to(device)

            output = model(input)

            loss = loss_fn(output, target)
            validation_loss += loss.item()

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
    # model = SimpleConvolutionalNetwork().to(device)

    model = UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=4,
        channels=[16, 32, 64, 128],
        strides=[2, 2, 2, 2],
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=2,
    ).to(device)

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
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--train_loss', type=bool, default=True, help='Print train loss or not')
    _args = parser.parse_args()

    main(_args)
