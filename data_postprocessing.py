import csv
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt


def process_image(row):
    # Extract image name and pixel values
    image_name = row[0]
    pixel_values = np.array(row[1:], dtype=np.uint8)

    # Reshape the flattened pixel values to a 128x128 array
    input_array = pixel_values.reshape(128, 128)

    # Convert to a binary image (0 for background, 1 for the object)
    binary_image = np.where(input_array > 0, 1, 0).astype(np.uint8)

    # Count occurrences of each value and find the most common one
    unique_values, counts = np.unique(input_array[input_array > 0], return_counts=True)

    if len(unique_values) == 0:
        # If there are no non-zero values, set most_common_value to 0
        most_common_value = 0
    else:
        most_common_value = unique_values[np.argmax(counts)]

    # Define the kernel for morphological operations
    if most_common_value == 1:
        kernel = np.ones((9, 9), np.uint8)
    elif most_common_value == 2:
        kernel = np.ones((5, 5), np.uint8)
    else:
        kernel = np.ones((1, 1), np.uint8)

    # Apply closing operation
    closing_result = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Multiply all elements by the most common value
    result_image = closing_result * most_common_value

    # Flatten the result image
    result_flattened = result_image.flatten()

    plt.subplot(1, 2, 1)
    plt.imshow(input_array, cmap='gray')
    plt.title('Input Image')

    plt.subplot(1, 2, 2)
    plt.imshow(closing_result, cmap='gray')
    plt.title('Closing Result')

    plt.show()

    time.sleep(2.5)

    # Construct the output row
    output_row = [image_name] + result_flattened.tolist()

    return output_row


# Load data from CSV
csv_path = "submission_plusplus.csv"

with open(csv_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    rows = list(reader)

# Remove the header row
header = rows[0]
data = rows[1:]

# Apply processing to each row of the data
processed_data = [process_image(row) for row in data]

# Create a new CSV file with the processed data
result_csv_path = "closed_submission_plusplus.csv"
with open(result_csv_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write header row
    writer.writerow(header)

    # Write processed data
    writer.writerows(processed_data)

print("Processing complete. Results saved to:", result_csv_path)
