import os

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical


def process_directory(directory, label, data_type):
    data = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpeg', '.jpg', '.png')):
                filepath = os.path.join(subdir, file)
                sublabel = 'Viral' if 'virus' in file else ('Bacterial' if 'bacteria' in file else 'Unknown')
                data.append((filepath, label, sublabel, data_type))
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
    label = to_categorical(label_map[label], num_classes=len(label_map))
    return image, label


def create_dataset(df, label_map, batch_size=32):
    filepaths = df['filepath'].values
    labels = df['label'].values

    def generator():
        for filepath, label in zip(filepaths, labels):
            yield preprocess_image(filepath, label, label_map)

    dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(150, 150, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(len(label_map),), dtype=tf.float32)))

    dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
