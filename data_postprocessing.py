import csv
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt


def process_image(row):
    image_name = row[0]
    pixel_values = np.array(row[1:], dtype=np.uint8)

    input_array = pixel_values.reshape(128, 128)
    binary_image = np.where(input_array > 0, 1, 0).astype(np.uint8)
    unique_values, counts = np.unique(input_array[input_array > 0], return_counts=True)

    if len(unique_values) == 0:
        most_common_value = 0
    else:
        most_common_value = unique_values[np.argmax(counts)]

    if most_common_value == 1:
        kernel = np.ones((3, 3), np.uint8)
    elif most_common_value == 2:
        kernel = np.ones((3, 3), np.uint8)
    else:
        kernel = np.ones((1, 1), np.uint8)

    closing_result = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    if most_common_value == 2:
        kernel = np.ones((3, 3), np.uint8)
        closing_result = cv2.erode(closing_result, kernel, iterations=2)
        closing_result = cv2.dilate(closing_result, kernel, iterations=2)

    result_image = closing_result * most_common_value
    result_flattened = result_image.flatten()

    # if most_common_value == 2:
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(input_array, cmap='gray')
    #     plt.title(image_name)
    #
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(closing_result, cmap='gray')
    #     plt.title('Closing Result')
    #
    #     plt.show()
    #
    #     time.sleep(3)

    output_row = [image_name] + result_flattened.tolist()
    return output_row


# Load data from CSV
csv_path = "voting/submission_094867.csv"

with open(csv_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    rows = list(reader)

header = rows[0]
data = rows[1:]

processed_data = [process_image(row) for row in data]
result_csv_path = "final_submissions_5.csv"

with open(result_csv_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(processed_data)

print("Processing complete. Results saved to:", result_csv_path)
