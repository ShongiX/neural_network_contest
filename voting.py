import os
import csv
from collections import defaultdict


def process_csv_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    data = defaultdict(lambda: defaultdict(list))

    # Reading data from CSV files
    for file_name in files:
        with open(os.path.join(directory, file_name), 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader)
            for row in csvreader:
                for col_idx, value in enumerate(row[1:]):
                    data[row[0]][col_idx].append(int(value))

    # Processing data to get the majority count
    majority_count = {}
    for row_id, columns in data.items():
        for col_idx, values in columns.items():
            count = sum(1 for val in values if val != 0)
            majority_count[(row_id, col_idx)] = 1 if count > len(files) / 2 else 0

    # Writing the majority count to a new CSV file
    output_filename = 'majority_vote_submission_binary_segment.csv'
    with open(output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        for row_id in sorted(data.keys()):
            row_values = [row_id] + [majority_count.get((row_id, col), 0) for col in range(16384)]
            csvwriter.writerow(row_values)

    print(f"Processed data written to {output_filename}")


def classify():
    class_index_dict = {}
    with open('classification.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            image_name, class_index = row
            class_index_dict[image_name] = int(class_index) - 1

    with open('majority_vote_submission_binary_segment.csv', newline='') as csvfile:
        with open('majority_vote_classification.csv', 'w', newline='') as output_csvfile:
            reader = csv.reader(csvfile)
            writer = csv.writer(output_csvfile)

            header = next(reader, None)
            writer.writerow(header)

            for row in reader:
                image_name = row[0]
                row_values = [int(val) for val in row[1:]]
                new_row_values = [i * (int(class_index_dict[image_name]) + 1) for i in row_values]
                writer.writerow([image_name] + new_row_values)

    print("Classification complete. Results saved to: majority_vote_classification.csv")


process_csv_files('voting')
classify()
