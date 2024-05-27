import os

import pandas as pd


def process_directory(directory, label):
    data = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpeg') or file.endswith('.jpg') or file.endswith('.png'):
                filepath = os.path.join(subdir, file)
                # Déduire le sous-label à partir du nom de fichier
                if 'virus' in file:
                    sublabel = 'Viral'
                elif 'bacteria' in file:
                    sublabel = 'Bacterial'
                else:
                    sublabel = 'Unknown'
                data.append((filepath, label, sublabel))
    return data


def generate_labeled_data(data_dir):
    data = []
    data += process_directory(os.path.join(data_dir, 'train/NORMAL'), 'Healthy')
    data += process_directory(os.path.join(data_dir, 'train/PNEUMONIA'), 'Pneumonia')
    data += process_directory(os.path.join(data_dir, 'val/NORMAL'), 'Healthy')
    data += process_directory(os.path.join(data_dir, 'val/PNEUMONIA'), 'Pneumonia')
    data += process_directory(os.path.join(data_dir, 'test/NORMAL'), 'Healthy')
    data += process_directory(os.path.join(data_dir, 'test/PNEUMONIA'), 'Pneumonia')

    df = pd.DataFrame(data, columns=['filepath', 'label', 'sublabel'])
    return df
