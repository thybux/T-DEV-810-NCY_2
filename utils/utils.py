# utils/utils.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data(data_dir, image_size, batch_size):
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="binary",
    )
    return generator
