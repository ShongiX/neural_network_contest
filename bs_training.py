import argparse
import random
import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from monai.networks.nets import BasicUNetPlusPlus
from torch import nn
from fastai.losses import DiceLoss, FocalLoss
from bs_data_preprocessing import load_data, create_image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter


def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return "%dh %02dm %02ds" % (hours, minutes, seconds)
    elif minutes > 0:
        return "%dm %02ds" % (minutes, seconds)
    else:
        return "%ds" % seconds


def train(train_dataloader, model, optimizer, loss_fn, device, metric, focal, cross_entropy_weight, dice_weight, focal_weight):
    for batch, (input, target) in enumerate(train_dataloader):
        input = input.to(device)
        target = target.to(device)

        output = model(input)[0]

        cross_entropy_loss = loss_fn(output, target)
        dice_loss = metric(output.permute(0, 2, 3, 1).contiguous().view(-1, output.size(1)), target.view(-1))
        focal_loss = focal(output, target)

        loss = cross_entropy_weight*cross_entropy_loss + dice_weight*dice_loss + focal_weight*focal_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # with torch.no_grad():
        #     if batch % 20 == 0:
        #         plt.subplot(1, 2, 1)
        #         plt.imshow(create_image(target[0, :, :].cpu().detach().numpy()))
        #         plt.subplot(1, 2, 2)
        #         output = torch.argmax(output, dim=1)
        #         plt.imshow(create_image(output[0, :, :].cpu().detach().numpy()))
        #         plt.show()


def validate(validation_dataloader, model, loss_fn, device, metric, focal, writer, epoch, cross_entropy_weight, dice_weight, focal_weight):
    cross_entropy_loss = 0
    dice_loss = 0
    focal_loss = 0

    with torch.no_grad():
        for batch, (input, target) in enumerate(validation_dataloader):
            input = input.to(device)
            target = target.to(device)

            output = model(input)[0]

            loss = loss_fn(output, target)
            cross_entropy_loss += loss.item()

            dice_loss += metric(output.permute(0, 2, 3, 1).contiguous().view(-1, output.size(1)), target.view(-1))
            focal_loss += focal(output, target)

    writer.add_scalar('Validation/CrossEntropyLoss', cross_entropy_loss, epoch)
    writer.add_scalar('Validation/DiceLoss', dice_loss, epoch)
    writer.add_scalar('Validation/FocalLoss', focal_loss, epoch)
    return cross_entropy_weight * cross_entropy_loss / len(validation_dataloader), dice_weight * dice_loss.item() / len(validation_dataloader), focal_weight * focal_loss.item() / len(validation_dataloader)


def main(args):
    start_time = time.time()
    set_seeds()

    writer = SummaryWriter(comment="BS_UNetPlusPlus")
    train_dataloader, validation_dataloader = load_data(args.class_index)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BasicUNetPlusPlus(
        spatial_dims=2,
        in_channels=2,
        out_channels=2,
        # features=[128, 256, 512, 512, 1024, 128],
        features=[305, 85, 261, 322, 213, 128]
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    metric = DiceLoss(axis=1, reduction='mean')
    focal = FocalLoss(gamma=2)

    cross_entropy_weight = 0.10204594423842461
    dice_weight = 0.10363852143665185
    focal_weight = 0.3169858253714459

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Training
    epochs = args.epochs
    best_test_loss = float('inf')
    cross_entropy_loss_list = []
    dice_loss_list = []
    focal_loss_list = []
    validation_loss_list = []

    for epoch in range(epochs):
        train(train_dataloader, model, optimizer, loss_fn, device, metric, focal, cross_entropy_weight, dice_weight, focal_weight)
        cross_entropy_loss, dice_loss, focal_loss = validate(validation_dataloader, model, loss_fn, device, metric, focal, writer, epoch, cross_entropy_weight, dice_weight, focal_weight)
        cross_entropy_loss_list.append(cross_entropy_loss)
        dice_loss_list.append(dice_loss)
        focal_loss_list.append(focal_loss)
        validation_loss = cross_entropy_weight*cross_entropy_loss + dice_weight*dice_loss + focal_weight*focal_loss
        validation_loss_list.append(validation_loss)

        if validation_loss < best_test_loss:
            best_test_loss = validation_loss
            torch.save(model.state_dict(), 'bs_' + str(args.class_index) + '.pth')

        scheduler.step(validation_loss)

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 1e-10:
            break

        writer.add_scalar('BestLoss', best_test_loss, epoch)

        print(f"Epoch: {epoch}, Validation loss: {validation_loss:>8f}, CrossEntropy loss: {cross_entropy_loss:>8f}, Dice loss: {dice_loss:>8f}, Focal loss: {focal_loss:>8f}, Elapsed time: {format_time(time.time() - start_time)}")

    writer.close()

    # Plot results
    x_axis = np.linspace(0, len(validation_loss_list), len(validation_loss_list))
    plt.plot(x_axis, validation_loss_list, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Loss vs Epochs')
    plt.show()

    print(
        f"Best test loss: {best_test_loss:>8f}, Elapsed time: {format_time(time.time() - start_time)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kaggle competition.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--train_loss', type=bool, default=True, help='Print train loss or not')
    parser.add_argument('--class_index', type=int, default=0, help='Class index to train the model')
    _args = parser.parse_args()

    main(_args)
