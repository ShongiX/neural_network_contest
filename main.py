import os
import numpy as np
import torch
import torch.nn as nn
from matplotlib import image as mpimg, pyplot as plt
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms


def create_segmentation_mask(rgb_image):
    # mask = np.zeros((128, 128), dtype=np.uint8)
    #
    # if rgb_image.shape == (128, 128, 4):
    #     rgb_image = rgb_image[:, :, :3]
    #
    # red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    # green = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    # blue = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    #
    # image_type = None
    # image_type_save = None
    #
    # for y in range(128):
    #     for x in range(128):
    #         pixel = rgb_image[y, x]
    #
    #         if np.array_equal(pixel, red):
    #             image_type = 0
    #         elif np.array_equal(pixel, green):
    #             image_type = 1
    #         elif np.array_equal(pixel, blue):
    #             image_type = 2
    #
    #         if image_type is not None:
    #             mask[y, x] = 1
    #             image_type_save = image_type
    #         image_type = None
    # return mask, image_type_save




class CompetitionDataset(Dataset):
    def __init__(self, img_dir, transform=None, validate=False):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.bmp')]
        self.validate = validate

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        label = os.path.splitext(image_name)[0]

        bmp_image = np.array(mpimg.imread(os.path.join(self.img_dir, f"{label}.bmp"))).astype(np.float32)
        tiff_image = np.array(mpimg.imread(os.path.join(self.img_dir, f"{label}.tiff"))).astype(np.float32)

        image = np.stack((tiff_image, bmp_image), axis=-1)

        if self.transform:
            image = self.transform(image)

        if self.validate:
            return image.reshape(2, 128, 128)

        png_image = np.array(mpimg.imread(os.path.join(self.img_dir, f"{label}_target.png"))).astype(np.float32)
        mask, type = create_segmentation_mask(png_image)
        return image, mask, type


def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    train_dataset = CompetitionDataset(img_dir='train_data', transform=transform, validate=False)

    validation_split = 0.2
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    batch_size = 5

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, test_loader


class SegmentationClassificationNetwork(nn.Module):
    def __init__(self):
        super(SegmentationClassificationNetwork, self).__init__()

        # Define the common initial layers
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Define the segmentation head
        self.segmentation_head = nn.Conv2d(64, 1, kernel_size=1)
        self.classification_head = nn.Linear(64 * 128 * 128, 3)

    def forward(self, x):
        # Common layers
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)

        # Segmentation head
        segmentation_map = self.segmentation_head(x)
        segmentation_output = nn.Hardsigmoid()(segmentation_map)

        # Classification head
        x = x.view(x.size(0), -1)  # Flatten the feature map
        x = x.view(x.size(0), -1)
        classification_output = self.classification_head(x)

        return segmentation_output, classification_output


def train(train_dataloader, model, optimizer, segmentation_loss_fn, classification_loss_fn, device):
    model.eval()

    for batch, (x, s, c) in enumerate(train_dataloader):
        x = x.to(device)
        s = s.to(device).to(torch.float32).squeeze()
        c = c.to(device)

        segmentation, classification = model(x)
        segmentation = segmentation.squeeze()

        loss = segmentation_loss_fn(segmentation.squeeze(), s) + classification_loss_fn(classification, c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(x)
        print(f"loss: {loss:>7f}  [{current:>5d}/{len(train_dataloader.dataset):>5d}]")


def validate(validation_dataloader, model, segmentation_loss_fn, classification_loss_fn, device):
    validation_loss = 0

    for batch, (x, s, c) in enumerate(validation_dataloader):
        x = x.to(device)
        s = s.to(device).to(torch.float32).squeeze()
        c = c.to(device)

        segmentation, classification = model(x)
        segmentation = segmentation.squeeze()

        validation_loss += segmentation_loss_fn(segmentation.squeeze(), s) + classification_loss_fn(classification, c)

        if batch % 20 == 0:
            loss, current = validation_loss, batch * len(x)
            print(f"validation loss: {loss:>7f}  [{current:>5d}/{len(validation_dataloader.dataset):>5d}]")

    return validation_loss / len(validation_dataloader)


def main():
    train_dataloader, test_dataloader = load_data()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    model = SegmentationClassificationNetwork().to(device)
    segmentation_criterion = nn.BCELoss()
    classification_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    epochs = 3

    loss_list = []
    best_test_loss = float('inf')

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, optimizer, segmentation_criterion, classification_criterion, device)
        loss = validate(test_dataloader, model, segmentation_criterion, classification_criterion, device)
        loss_list.append(loss)

        if loss < best_test_loss:
            best_test_loss = loss
            torch.save(model.state_dict(), 'model.pth')
    print("Done!")

    x_axis = np.linspace(0, len(loss_list), len(loss_list))
    plt.plot(x_axis, loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.show()


if __name__ == '__main__':
    main()
