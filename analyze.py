import torch
from matplotlib import pyplot as plt
from monai.networks.nets import BasicUNetPlusPlus
from torch import nn
from fastai.losses import DiceLoss, FocalLoss
from bs_data_preprocessing import load_data, create_image
from torch.utils.data import DataLoader


def visualize_worst_performing_images(validation_dataloader, model, device, dice):
    losses = []

    with torch.no_grad():
        for batch, (input, target) in enumerate(validation_dataloader):
            input = input.to(device)
            target = target.to(device)

            output = model(input)[0]

            dice_loss = dice(output.permute(0, 2, 3, 1).contiguous().view(-1, output.size(1)), target.view(-1))

            losses.append((dice_loss.item(), input, target, output))

    # Sort by losses to get the worst-performing images
    losses.sort(key=lambda x: x[0], reverse=True)

    # Visualize top N worst-performing images
    num_images_to_visualize = 5  # Change this number to visualize more or fewer images
    for i in range(num_images_to_visualize):
        _, input_img, target_img, output_img = losses[i]
        plt.subplot(1, 2, 1)
        plt.imshow(create_image(target_img[0, :, :].cpu().detach().numpy()))
        plt.title('Ground Truth')
        plt.subplot(1, 2, 2)
        output_img = torch.argmax(output_img, dim=1)
        plt.imshow(create_image(output_img[0, :, :].cpu().detach().numpy()))
        plt.title('Predicted')
        plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_0 = BasicUNetPlusPlus(spatial_dims=2, in_channels=2, out_channels=2, features=[305, 85, 261, 322, 213, 128]).to(device)
    model_0.load_state_dict(torch.load('bs_0.pth'))

    model_1 = BasicUNetPlusPlus(spatial_dims=2, in_channels=2, out_channels=2, features=[305, 85, 261, 322, 213, 128]).to(device)
    model_1.load_state_dict(torch.load('bs_1.pth'))

    model_2 = BasicUNetPlusPlus(spatial_dims=2, in_channels=2, out_channels=2, features=[305, 85, 261, 322, 213, 128]).to(device)
    model_2.load_state_dict(torch.load('bs_2.pth'))

    # Define loss function, optimizer, and other necessary components
    dice = DiceLoss()

    # Load validation data
    validate_dataloader1 = load_data(class_index=0)[1]
    validate_dataloader2 = load_data(class_index=1)[1]
    validate_dataloader3 = load_data(class_index=2)[1]

    # Visualize worst-performing images for each model
    visualize_worst_performing_images(validate_dataloader1, model_0, device, dice)
    visualize_worst_performing_images(validate_dataloader2, model_1, device, dice)
    visualize_worst_performing_images(validate_dataloader3, model_2, device, dice)


if __name__ == '__main__':
    main()
