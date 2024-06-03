from tensorflow.keras import layers, models


def model_ia(train_dataset, val_dataset, test_dataset):
    model = models.Sequential([
        layers.InputLayer(input_shape=(150, 150, 3)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),  # Ajout d'une couche Dropout

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),  # Ajout d'une couche Dropout

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),  # Ajout d'une couche Dropout

        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Ajout d'une couche Dropout

        layers.Dense(2, activation='softmax')  # Assuming 2 classes: Healthy and Pneumonia
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)

    val_loss, val_acc = model.evaluate(val_dataset)

    return model, history, val_loss, val_acc
