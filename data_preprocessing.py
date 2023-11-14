import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image


def create_image(output_mask):
    output_image = np.zeros((output_mask.shape[0], output_mask.shape[1], 3))
    for i in range(output_mask.shape[0]):
        for j in range(output_mask.shape[1]):
            if output_mask[i, j] == 0:
                output_image[i, j, :] = [0, 0, 0]
            elif output_mask[i, j] == 1:
                output_image[i, j, :] = [1, 0, 0]
            elif output_mask[i, j] == 2:
                output_image[i, j, :] = [0, 1, 0]
            elif output_mask[i, j] == 3:
                output_image[i, j, :] = [0, 0, 1]
    return output_image


def create_mask(target_image):
    target_mask = np.zeros_like(target_image[0, :, :])
    for i in range(target_image.shape[1]):
        for j in range(target_image.shape[2]):
            if all(target_image[:, i, j] == [0, 0, 0]):
                target_mask[i, j] = 0
            elif all(target_image[:, i, j] == [255, 0, 0]):
                target_mask[i, j] = 1
            elif all(target_image[:, i, j] == [0, 255, 0]):
                target_mask[i, j] = 2
            elif all(target_image[:, i, j] == [0, 0, 255]):
                target_mask[i, j] = 3
    return target_mask


class CompetitionDataset(Dataset):
    def __init__(self, img_dir, test=False):
        self.img_dir = img_dir
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.bmp')]
        self.test = test

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        label = os.path.splitext(image_name)[0]

        bmp_image = np.array(Image.open(os.path.join(self.img_dir, f"{label}.bmp"))).astype(
            np.float32)
        tiff_image = np.array(Image.open(os.path.join(self.img_dir, f"{label}.tiff"))).astype(
            np.float32)
        input_image = np.stack((tiff_image, bmp_image, np.zeros_like(tiff_image)), axis=-1)
        input_image = input_image.transpose((2, 0, 1))
        input_image = torch.FloatTensor(input_image)

        if self.test:
            return input_image

        target_image = np.array(Image.open(os.path.join(self.img_dir, f"{label}_target.png"))).astype(np.float32)
        if target_image.shape == (128, 128, 4):
            target_image = target_image[:, :, :3]
        target_image = target_image.transpose((2, 0, 1))
        target_mask = create_mask(target_image)
        target_mask = torch.FloatTensor(target_mask)

        return input_image.float(), target_mask.float()


def load_data():
    train_dataset = CompetitionDataset(img_dir='train_data', test=False)
    validate_dataset = CompetitionDataset(img_dir='validate_data', test=False)

    batch_size = 5

    _train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    _validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size)

    return _train_dataloader, _validate_dataloader


if __name__ == '__main__':
    train_dataloader, validate_dataloader = load_data()
    for i, (input, target) in enumerate(train_dataloader):
        plot_input = input[0, :, :, :].numpy().transpose((1, 2, 0))
        plot_target = target[0, :, :].numpy()

        plot_input = Image.fromarray(plot_input.astype(np.uint8))
        plot_target = Image.fromarray(plot_target.astype(np.uint8))

        plt.subplot(1, 2, 1)
        plt.imshow(plot_input)
        plt.subplot(1, 2, 2)
        plt.imshow(plot_target)
        plt.show()

        break


