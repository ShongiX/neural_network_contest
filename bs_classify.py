import csv
import numpy as np


def process_image(row):
    image_name = row[0]
    pixel_values = np.array(row[1:], dtype=np.uint8)

    input_array = pixel_values.reshape(128, 128)
    unique_values, counts = np.unique(input_array[input_array > 0], return_counts=True)

    if len(unique_values) == 0:
        most_common_value = 0
    else:
        most_common_value = unique_values[np.argmax(counts)]

    output_row = [image_name] + [most_common_value]

    return output_row


csv_path = "submission_plusplusplus.csv"

with open(csv_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    rows = list(reader)

header = rows[0]
data = rows[1:]
processed_data = [process_image(row) for row in data]

result_csv_path = "classification.csv"
with open(result_csv_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(processed_data)

print("Processing complete. Results saved to:", result_csv_path)
