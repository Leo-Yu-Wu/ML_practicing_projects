# ----------------------------------------------------------------------------
# File: create_data.py
# Description: This file create datasets for a machine learning model that classify points based on their feature
# Author: Leo Wu
# Date Created: 2025-05-21
# Last Modified: 2025-05-21
# Version: 1.0
# Dependencies: numpy
# ----------------------------------------------------------------------------

import numpy as np
import csv
import os
# Set random seed for reproducibility
np.random.seed(0)

def create_data(filename: str, num_data: int, noise_rate: float = 0.0):
    """
    Generate datas based on the following rule:
        Given values for Feature 1, Feature 2, and Feature 3, where:
         - Feature 1 (x): Between 0 and 100
         - Feature 2 (y): Between 0 and 100
         - Feature 3 (z): Between 0 and 50

        Points where Feature 3 (z) is greater than 30 are labeled as Class B.
        Points where Feature 3 (z) is less than or equal to 30 and Feature 1 (x) is greater than 30 are labeled as Class B.
        Otherwise, the points are labeled as Class A.
    :param filename: the name of the file, example: train.csv
    :param num_data: number of data points to generate
    :noise_rate: rate of noise where the data points are mis-labeled, defalut: 0.0
    """
    # ensure proper file name
    if not filename.endswith('.csv'):
        filename = filename + '.csv'
    output_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.join(output_folder, filename)
    # create a filename.csv file, override if one already exist
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['x', 'y', 'z', 'label'])
        while num_data > 0:
            # default label: A
            label = 'A'
            x = np.random.randint(0, 100)
            y = np.random.randint(0, 100)
            z = np.random.randint(0, 50)
            noise = np.random.rand()
            # if z > 30, label is set to B
            if z > 30:
                label = 'B'
            # if z is less than or equal to 30 and x is greater than 30 label is set to B
            elif z <= 30 < x:
                label = 'B'

            # add noise
            if noise < noise_rate:
                if label == 'A':
                    label = 'B'
                elif label == 'B':
                    label = 'A'

            # write row
            writer.writerow([x, y, z, label])
            num_data -= 1

    # finish generating
    print(f'Complete generating {filename} with noise {noise_rate * 100}%')


if __name__ == '__main__':
    create_data('train.csv', 1000, noise_rate=0.005)
    create_data('test.csv', 200, noise_rate=0)