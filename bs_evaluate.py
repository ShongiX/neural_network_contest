import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from monai.networks.nets import BasicUNetPlusPlus
from torch.utils.data import DataLoader
import csv

from bs_data_preprocessing import BinarySegmentDataset, create_image


def main(args):
    start_time = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class_index_dict = {}
    with open('classification.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            image_name, class_index = row
            class_index_dict[image_name] = int(class_index)-1

    models = {}
    for class_index in range(3):
        model_path = f'bs_{class_index}_seed{args.version}.pth'
        model = BasicUNetPlusPlus(
            spatial_dims=2,
            in_channels=2,
            out_channels=2,
            features=[305, 85, 261, 322, 213, 128]
        ).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        models[class_index] = model

    test_dataset = BinarySegmentDataset(img_dir='test_data', class_index=-1, test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    with open('submission_binary_segment_' + str(args.version) + '.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        header_row = ['id'] + [str(i) for i in range(1, 16385)]
        csvwriter.writerow(header_row)

        with torch.no_grad():
            for batch, (input, label) in enumerate(test_dataloader):
                image_name = label[0]
                class_index = class_index_dict[image_name]
                model = models[class_index]

                input = input.to(device)

                output = model(input)[0]
                output = torch.argmax(output, dim=1).cpu().detach().numpy()
                output *= (class_index + 1)

                # plt.subplot(1, 2, 1)
                # plot_input = Image.fromarray(input[0, :, :, :].cpu().detach().numpy().transpose((1, 2, 0)).astype(np.uint8))
                # plt.imshow(plot_input)
                # plt.subplot(1, 2, 2)
                # plt.imshow(create_image(output[0, :, :]))
                # plt.show()
                #
                # time.sleep(3)

                output = output.flatten()
                data_row = [image_name] + [str(num) for num in output]
                csvwriter.writerow(data_row)

    print("Done! (%s)" % (time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--version', type=int, default=0, help='Version of the model')
    args = parser.parse_args()

    main(args)
