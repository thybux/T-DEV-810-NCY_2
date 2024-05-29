import os

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical


def process_directory(directory, label, type):
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
                data.append((filepath, label, sublabel, type))
    return data


def generate_labeled_data(data_dir):
    data = []
    data += process_directory(os.path.join(data_dir, 'train/NORMAL'), 'Healthy', 'train')
    data += process_directory(os.path.join(data_dir, 'train/PNEUMONIA'), 'Pneumonia', 'train')
    data += process_directory(os.path.join(data_dir, 'val/NORMAL'), 'Healthy', 'val')
    data += process_directory(os.path.join(data_dir, 'val/PNEUMONIA'), 'Pneumonia', 'val')
    data += process_directory(os.path.join(data_dir, 'test/NORMAL'), 'Healthy', 'test')
    data += process_directory(os.path.join(data_dir, 'test/PNEUMONIA'), 'Pneumonia', 'test')

    df = pd.DataFrame(data, columns=['filepath', 'label', 'sublabel', 'type'])
    return df


def preprocess_image(filepath, label, label_map, target_size=(150, 150)):
    image = load_img(filepath, target_size=target_size)
    image = img_to_array(image)
    image = image / 255.0
    label = to_categorical(label, num_classes=len(label_map))
    return image, label


def create_dataset(df, batch_size=32):
    filepaths = df['filepath'].values
    labels = df['label'].values

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.map(lambda x, y: tf.py_function(preprocess_image, [x, y], [tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
