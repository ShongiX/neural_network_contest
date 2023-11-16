import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from monai.networks.nets import UNet, BasicUNetPlusPlus
from torch.utils.data import DataLoader
import csv

from data_preprocessing import CompetitionDataset, create_image


def main():
    start_time = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    model = BasicUNetPlusPlus(
        spatial_dims=2,
        in_channels=3,
        out_channels=4,
        features=[64, 128, 256, 512, 1024, 128],
    ).to(device)
    model.load_state_dict(torch.load('unetplusplus.pth'))
    test_dataset = CompetitionDataset(img_dir='test_data', test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    with open('submission_plusplus.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')

        header_row = ['id'] + [str(i) for i in range(1, 16385)]
        csvwriter.writerow(header_row)

        with torch.no_grad():
            for batch, (input, label) in enumerate(test_dataloader):
                input = input.to(device)

                output = model(input)[0]

                output = torch.argmax(output, dim=1)

                if batch % 1 == 0:
                    plt.subplot(1, 2, 1)
                    plot_input = Image.fromarray(input[0, :, :, :].cpu().detach().numpy().transpose((1, 2, 0)).astype(np.uint8))
                    plt.imshow(plot_input)
                    plt.subplot(1, 2, 2)
                    plt.imshow(create_image(output[0, :, :].cpu().detach().numpy()))
                    plt.show()

                time.sleep(2.5)

                output = output.cpu().detach().numpy()
                output = output.flatten()

                label_str = label[0]

                data_row = [label_str] + [str(num) for num in output]
                csvwriter.writerow(data_row)

    print("Done! (%s)" % (time.time() - start_time))


if __name__ == "__main__":
    main()
