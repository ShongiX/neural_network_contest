import os
import numpy as np
import torch
from matplotlib import image as mpimg
from monai.transforms import ScaleIntensity, RandFlip, RandRotate, RandZoom
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


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

        bmp_image = np.array(mpimg.imread(os.path.join(self.img_dir, f"{label}.bmp"))).astype(np.float32)
        tiff_image = np.array(mpimg.imread(os.path.join(self.img_dir, f"{label}.tiff"))).astype(np.float32)
        input_image = np.stack((tiff_image, bmp_image), axis=-1)
        input_image = input_image.reshape(2, 128, 128)
        # input_image = transforms.Compose([
        #     ToTensor(),
        #     ScaleIntensity(),
        #     RandFlip(prob=0.5),
        #     RandRotate(range_x=np.pi / 8, prob=0.5),
        #     RandZoom(min_zoom=0.9, max_zoom=1.1)
        # ])(input_image)
        input_image = torch.Tensor(input_image)

        if self.test:
            return input_image

        target_image = np.array(mpimg.imread(os.path.join(self.img_dir, f"{label}_target.png"))).astype(np.float32)
        if target_image.shape == (128, 128, 4):
            target_image = target_image[:, :, :3]
        target_image = target_image.reshape(3, 128, 128)
        target_image = torch.Tensor(target_image)

        input_image = input_image.reshape(2, 128, 128)
        target_image = target_image.reshape(3, 128, 128)

        return input_image, target_image


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
        plot_input = input[0, :, :, :].numpy().reshape(128, 128, 2)
        plot_target = target[0, :, :, :].numpy().reshape(128, 128, 3)

        plt.subplot(1, 2, 1)
        plt.imshow(plot_input[:, :, 0])
        plt.subplot(1, 2, 2)
        plt.imshow(plot_target)
        plt.show()
        plt.title("Sample from train dataset")
        break
