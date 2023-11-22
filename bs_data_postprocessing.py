import csv
import numpy as np
import cv2
from pytorchcrf import CRF


def read_csv(csv_path):
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
    header = rows[0]
    data = rows[1:]
    return header, data


def process_image(row):
    pixel_values = np.array(row[1:], dtype=np.uint8).reshape(128, 128)
    binary_image = np.where(pixel_values > 0, 1, 0).astype(np.uint8)
    unique_values, counts = np.unique(pixel_values[pixel_values > 0], return_counts=True)
    most_common_value = unique_values[np.argmax(counts)] if len(unique_values) > 0 else 0

    if most_common_value == 1:
        kernel = np.ones((7, 7), np.uint8)
    elif most_common_value == 2:
        kernel = np.ones((5, 5), np.uint8)
    else:
        kernel = np.ones((1, 1), np.uint8)

    closing_result = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return row[0], closing_result.flatten()


def apply_crf(images):
    crf = CRF(num_tags=1, batch_first=True)
    crf_input = np.expand_dims(images, axis=1)
    crf_output = crf.forward(crf_input)
    return crf_output.squeeze(axis=1).astype(np.uint8)


def write_csv(header, output_data, result_csv_path):
    with open(result_csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(output_data)
    print("Processing complete. Results saved to:", result_csv_path)


def process_images_with_crf(csv_path, result_csv_path):
    header, data = read_csv(csv_path)
    processed_data = [process_image(row) for row in data]
    image_names, processed_images = zip(*processed_data)
    processed_images_array = np.array(processed_images)
    crf_output_flattened = apply_crf(processed_images_array)
    output_data = [[image_names[i]] + crf_output_flattened[i].tolist() for i in range(len(image_names))]
    write_csv(header, output_data, result_csv_path)


if __name__ == "__main__":
    csv_path = "submission_plusplusplus.csv"
    result_csv_path = "closed_crf_submission_plusplusplus.csv"
    process_images_with_crf(csv_path, result_csv_path)
