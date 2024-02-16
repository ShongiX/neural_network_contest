import argparse
import numpy as np
import optuna
import torch
from monai.networks.nets import BasicUNetPlusPlus
from torch import nn
from fastai.losses import DiceLoss, FocalLoss
from tqdm import tqdm

from bs_data_preprocessing import load_data, create_image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter


def set_seeds(seed=42):
    np.random.seed(seed)
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


train_dataloader = None
validation_dataloader = None
writer = None


def objective(trial):
    feature_1 = trial.suggest_int('feature_1', 64, 512, log=True)
    feature_2 = trial.suggest_int('feature_2', 64, 512, log=True)
    feature_3 = trial.suggest_int('feature_3', 64, 512, log=True)
    feature_4 = trial.suggest_int('feature_4', 64, 1024, log=True)
    feature_5 = trial.suggest_int('feature_5', 64, 1024, log=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BasicUNetPlusPlus(
        spatial_dims=2,
        in_channels=2,
        out_channels=2,
        features=[feature_1, feature_2, feature_3, feature_4, feature_5, 128],
    ).to(device)

    cross_entropy_weight = 0.10204594423842461
    dice_weight = 0.10363852143665185
    focal_weight = 0.3169858253714459

    loss_fn = nn.CrossEntropyLoss()
    metric = DiceLoss(axis=1, reduction='mean')
    focal = FocalLoss(gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    best_validation_loss = float('inf')

    epochs = 10
    for epoch in range(epochs):
        train(train_dataloader, model, optimizer, loss_fn, device, metric, focal, cross_entropy_weight, dice_weight, focal_weight)
        cross_entropy_loss, dice_loss, focal_loss = validate(validation_dataloader, model, loss_fn, device, metric, focal, writer, epoch, cross_entropy_weight, dice_weight, focal_weight)
        validation_loss = cross_entropy_weight*cross_entropy_loss + dice_weight*dice_loss + focal_weight*focal_loss

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

        scheduler.step(validation_loss)

    return best_validation_loss


def main(args):
    set_seeds()

    _writer = SummaryWriter(comment="BS_UNetPlusPlus")
    _train_dataloader, _validation_dataloader = load_data(args.class_index)
    global train_dataloader
    global validation_dataloader
    train_dataloader = _train_dataloader
    validation_dataloader = _validation_dataloader
    global writer
    writer = _writer

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())

    with tqdm(total=400) as progress_bar:
        def update_progress_bar(study, trial):
            progress_bar.set_description(
                f"Evaluating trial {trial.number}/{400}")
            progress_bar.update(1)

        study.optimize(objective, n_trials=400, callbacks=[update_progress_bar])

    writer.close()
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kaggle competition.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--class_index', type=int, default=0, help='Class index to train the model')
    _args = parser.parse_args()

    main(_args)
