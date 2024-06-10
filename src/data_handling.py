import os

import pandas as pd
import tensorflow as tf
from PIL import Image
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils import load_img
from tensorflow.keras.preprocessing.image import img_to_array
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


def preprocess_image_with_tensorflow(filepath, label, label_map, target_size=(150, 150)):
    # Charger l'image en niveaux de gris
    image = load_img(filepath, color_mode='grayscale')
    image_array = img_to_array(image)

    # Utiliser TensorFlow pour redimensionner l'image
    image_resized = tf.image.resize_with_pad(image_array, target_size[0], target_size[1])
    image_resized = image_resized / 255.0  # Normaliser les valeurs des pixels entre 0 et 1
    image_resized = tf.reshape(image_resized, (target_size[0], target_size[1], 1))  # Assurez-vous que l'image a 1 canal

    # Convertir l'étiquette en représentation catégorielle
    label = to_categorical(label_map[label], num_classes=len(label_map))

    return image_resized.numpy(), label


def create_dataset(df, label_map, batch_size=32):
    filepaths = df['filepath'].values
    labels = df['label'].values

    def generator():
        for filepath, label in zip(filepaths, labels):
            yield preprocess_image_with_tensorflow(filepath, label, label_map)

    dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(150, 150, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(len(label_map),), dtype=tf.float32)))

    dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def increase_data(data, save_dir, copy):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    for i, filepath in enumerate(data['filepath'].values):
        img = Image.open(filepath)
        img_array = img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape)
        # Générer des images augmentées
        for j, batch in enumerate(
                datagen.flow(img_array, batch_size=1, save_to_dir=save_dir, save_prefix=f'Healthy_{i}',
                             save_format='png')):
            if j >= copy:
                break
