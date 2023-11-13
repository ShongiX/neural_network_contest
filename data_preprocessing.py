import os
import numpy as np
from matplotlib import image as mpimg
from monai.transforms import ScaleIntensity, RandFlip, RandRotate, RandZoom
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor


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
        input_image = transforms.Compose([
            ToTensor(),
            ScaleIntensity(),
            RandFlip(prob=0.5),
            RandRotate(range_x=np.pi / 8, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1)
        ])(input_image)

        if self.test:
            return input_image

        target_image = np.array(mpimg.imread(os.path.join(self.img_dir, f"{label}_target.png"))).astype(np.float32)
        if target_image.shape == (128, 128, 4):
            target_image = target_image[:, :, :3]
        target_image = target_image.reshape(3, 128, 128)
        target_image = transforms.Compose([
            ToTensor()
        ])(target_image)

        return input_image, target_image


def load_data():
    train_dataset = CompetitionDataset(img_dir='train_data', test=False)
    validate_dataset = CompetitionDataset(img_dir='validate_data', test=False)

    batch_size = 5

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size)

    return train_dataloader, validate_dataloader