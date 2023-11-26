import argparse
import random
import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from monai.losses import HausdorffDTLoss, DiceFocalLoss, DiceCELoss
from monai.networks.nets import BasicUNetPlusPlus
from torch import nn
from fastai.losses import DiceLoss, FocalLoss
from bs_data_preprocessing import load_data, create_image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, args):
        self.epochs = args.epochs
        self.class_index = args.class_index
        self.version = args.version
        self.train_dataloader, self.validation_dataloader = load_data(self.class_index)

        self.cross_entropy_weight = 1
        self.dice_weight = 0
        self.hausdorff_weight = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter(comment="BS_UNetPlusPlus")

        self.model = self._initialize_model()
        self.cross = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        self.dice = DiceFocalLoss(include_background=False, to_onehot_y=True, softmax=True)
        self.hausdorff = HausdorffDTLoss(include_background=False, softmax=True, to_onehot_y=True)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-11)

        self.cross_entropy_loss_list = []
        self.dice_loss_list = []
        self.hausdorff_loss_list = []
        self.validation_loss_list = []
        self.best_test_loss = float('inf')

    def _initialize_model(self):
        return BasicUNetPlusPlus(
            spatial_dims=2,
            in_channels=2,
            out_channels=2,
            features=[305, 85, 261, 322, 213, 128]
        ).to(self.device)

    @staticmethod
    def set_seeds(seed=87):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def format_time(seconds):
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return "%dh %02dm %02ds" % (hours, minutes, seconds)
        elif minutes > 0:
            return "%dm %02ds" % (minutes, seconds)
        else:
            return "%ds" % seconds

    def train(self):
        start_time = time.time()
        for epoch in range(self.epochs):
            self._train_epoch(self.train_dataloader)
            cross_entropy_loss, dice_loss, hausdorff_loss = self._validate_epoch(self.validation_dataloader)
            validation_loss = cross_entropy_loss + dice_loss + hausdorff_loss
            self._update_loss_lists(cross_entropy_loss, dice_loss, hausdorff_loss, validation_loss)

            if validation_loss < self.best_test_loss:
                self.best_test_loss = validation_loss
                torch.save(self.model.state_dict(), 'bs_' + str(self.class_index) + '_seed' + str(self.version) + '.pth')

            self.scheduler.step(validation_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr < 1e-9:
                break

            self.writer.add_scalar('BestLoss', self.best_test_loss, epoch)
            self._print_epoch_results(epoch, start_time, validation_loss, cross_entropy_loss, dice_loss, hausdorff_loss)

        self.writer.close()
        self._plot_results()

        print(
            f"Best test loss: {self.best_test_loss:>8f}, Elapsed time: {self.format_time(time.time() - start_time)}")

    def _train_epoch(self, train_dataloader):
        self.model.train()
        for batch, (input, target) in enumerate(train_dataloader):
            input = input.to(self.device)
            target = target.to(self.device)
            output = self.model(input)[0]

            cross_entropy_loss = self.cross(output, target.unsqueeze(1))
            dice_loss = self.dice(output, target.unsqueeze(1))
            hausdorff_loss = self.hausdorff(output, target.unsqueeze(1))

            loss = self.cross_entropy_weight * cross_entropy_loss + self.dice_weight * dice_loss + self.hausdorff_weight * hausdorff_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _validate_epoch(self, validation_dataloader):
        self.model.eval()
        cross_entropy_loss = 0
        dice_loss = 0
        hausdorff_loss = 0
        with torch.no_grad():
            for batch, (input, target) in enumerate(validation_dataloader):
                input = input.to(self.device)
                target = target.to(self.device)
                output = self.model(input)[0]

                cross_entropy_loss += self.cross(output, target.unsqueeze(1))
                dice_loss += self.dice(output, target.unsqueeze(1))
                hausdorff_loss += self.hausdorff(output, target.unsqueeze(1))

        return self.cross_entropy_weight * cross_entropy_loss.item() / len(validation_dataloader), \
               self.dice_weight * dice_loss.item() / len(validation_dataloader), \
               self.hausdorff_weight * hausdorff_loss.item() / len(validation_dataloader)

    def _update_loss_lists(self, cross_entropy_loss, dice_loss, hausdorff_loss, validation_loss):
        self.cross_entropy_loss_list.append(cross_entropy_loss)
        self.dice_loss_list.append(dice_loss)
        self.hausdorff_loss_list.append(hausdorff_loss)
        self.validation_loss_list.append(validation_loss)

    def _print_epoch_results(self, epoch, start_time, validation_loss, cross_entropy_loss, dice_loss, hausdorff_loss):
        print(
            f"Epoch: {epoch}, Validation loss: {validation_loss:>8f}, CrossEntropy loss: {cross_entropy_loss:>8f}, "
            f"DiceFocal loss: {dice_loss:>8f}, Hausdorff loss: {hausdorff_loss:>8f}, "
            f"Elapsed time: {self.format_time(time.time() - start_time)}")

    def _plot_results(self):
        x_axis = np.linspace(0, len(self.validation_loss_list), len(self.validation_loss_list))
        plt.plot(x_axis, self.validation_loss_list, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Validation Loss')
        plt.title('Loss vs Epochs')
        plt.show()

    def main(self):
        # self.set_seeds()
        self.train_dataloader, self.validation_dataloader = load_data(self.class_index)
        self.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kaggle competition.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--class_index', type=int, default=0, help='Class index to train the model')
    parser.add_argument('--version', type=int, default=0, help='Version of the model')
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.main()
