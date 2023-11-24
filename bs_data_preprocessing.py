import os
import numpy as np
import torchio as tio
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


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
            else:
                target_mask[i, j] = 1
    return target_mask


def get_class_index(class_name):
    if class_name == 'fonalas':
        return 0
    elif class_name == 'sco':
        return 1
    elif class_name == 'trc':
        return 2


def add_gaussian_noise(image):
    noise = np.random.normal(0, 20, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


class BinarySegmentDataset(Dataset):
    def __init__(self, img_dir, class_index, test=False):
        self.img_dir = img_dir
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.bmp')]
        self.test = test
        self.transform = tio.Compose([
            tio.ZNormalization(),
        ])
        self.transform2 = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
        ])

        if not test:
            self.image_files = [f for f in self.image_files if get_class_index(f.split('_')[0]) == class_index]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        label = os.path.splitext(image_name)[0]

        bmp_image = np.array(Image.open(os.path.join(self.img_dir, f"{label}.bmp"))).astype(
            np.float32)
        tiff_image = np.array(Image.open(os.path.join(self.img_dir, f"{label}.tiff"))).astype(
            np.float32)
        input_image = np.stack((tiff_image, bmp_image), axis=-1)
        input_image = input_image.transpose((2, 0, 1))
        input_image = add_gaussian_noise(input_image)
        input_image = torch.FloatTensor(input_image)

        input_image = input_image[..., np.newaxis]
        input_image = self.transform(input_image)
        input_image = input_image[..., 0]

        if self.test:
            return input_image, label+'_target'

        target_image = np.array(Image.open(os.path.join(self.img_dir, f"{label}_target.png"))).astype(np.float32)
        if target_image.shape == (128, 128, 4):
            target_image = target_image[:, :, :3]
        target_image = target_image.transpose((2, 0, 1))
        target_mask = create_mask(target_image)
        target_mask = torch.Tensor(target_mask)

        concat = torch.cat((input_image, target_mask.unsqueeze(0)), dim=0)
        concat = self.transform2(concat)
        input_image = concat[:2, :, :]
        target_mask = concat[2, :, :]

        return input_image, target_mask.long()


def load_data(class_index=0):
    train_dataset = BinarySegmentDataset(img_dir='combined_data', class_index=class_index, test=False)
    validate_dataset = BinarySegmentDataset(img_dir='validate_data', class_index=class_index, test=False)

    batch_size = 5

    _train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    _validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size)

    return _train_dataloader, _validate_dataloader


if __name__ == '__main__':
    train_dataloader, validate_dataloader = load_data()
    for i, (input, target) in enumerate(train_dataloader):
        plot_input = input[0, :, :, :].numpy().transpose((1, 2, 0))
        plot_target = target[0, :, :].numpy()

        plot_input = Image.fromarray(plot_input[:, :, 1].astype(np.uint8))
        plot_target = Image.fromarray(plot_target.astype(np.uint8))

        plt.subplot(1, 2, 1)
        plt.imshow(plot_input)
        plt.subplot(1, 2, 2)
        plt.imshow(plot_target)
        plt.show()


