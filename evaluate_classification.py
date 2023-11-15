import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from monai.networks.nets import UNet
from torch import nn
from torch.utils.data import DataLoader
import csv

from torchvision.models import resnet152, ResNet152_Weights

from data_preprocessing import CompetitionDataset, create_image
from simple_convolution import SimpleConvolutionalNetwork


def main():
    start_time = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 5
    model = resnet152(weights=ResNet152_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    model = model.to(device)
    model.load_state_dict(torch.load('resnet.pth'))
    test_dataset = CompetitionDataset(img_dir='test_data', classify=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    total = 0
    correct = 0

    with open('classification.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')

        header_row = ['id', 'label']
        csvwriter.writerow(header_row)

        with torch.no_grad():
            for batch, (input, label, name) in enumerate(test_dataloader):
                input = input.to(device)
                label = label.to(device)

                output = model(input)

                output = torch.softmax(output, dim=1)

                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                for i in range(len(name)):
                    csvwriter.writerow([name[i], predicted[i].cpu().detach().numpy()+1])

    accuracy = 100 * correct / total
    print("Accuracy = " + str(accuracy) + "%" + ", Time = " + str(time.time() - start_time))


if __name__ == "__main__":
    main()
