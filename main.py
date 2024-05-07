# main.py
from models.models import create_model
from utils.utils import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model


def predict_image(image_path, model_path):
    model = load_model(
        model_path
    )  # Charge le modèle depuis le chemin du fichier modèle
    img = load_img(image_path, target_size=(150, 150), color_mode="rgb")
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalisation des pixels
    img_array = img_array.reshape(
        (1, 150, 150, 3)
    )  # Redimensionner l'array pour le modèle

    # Faire la prédiction
    prediction = model.predict(img_array)[0, 0]
    if prediction > 0.7:
        print(f"Prédiction: Pneumonie avec une confiance de {
              prediction*100:.2f}%")
    else:
        print(f"Prédiction: Normal avec une confiance de {
              (1 - prediction)*100:.2f}%")


def evaluate_model(model_path, test_dir, image_size):
    model = load_model(model_path)

    # Préparation du générateur de données
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    generator = datagen.flow_from_directory(
        test_dir,
        target_size=(image_size, image_size),
        batch_size=1,
        class_mode="binary",
        shuffle=False,
    )

    # Évaluation
    results = model.evaluate(generator, steps=len(generator))
    print(f"Loss: {results[0]}, Accuracy: {results[1]*100:.2f}%")

    # Prédiction et affichage des résultats
    generator.reset()
    predictions = model.predict(generator, steps=len(generator))
    for i, pred in enumerate(predictions):
        actual_class = "Pneumonie" if generator.classes[i] else "Normal"
        predicted_class = "Pneumonie" if pred > 0.5 else "Normal"
        print(f"Image: {os.path.basename(generator.filenames[i])} - Actual: {
              actual_class}, Predicted: {predicted_class}, Confidence: {max(pred, 1-pred)[0]*100:.2f}%")


def train_model():
    # Paramètres
    image_height, image_width = 150, 150
    batch_size = 20

    # Chargement des données
    train_data = load_data("datasets/chest_Xray/train",
                           image_height, batch_size)
    validation_data = load_data(
        "datasets/chest_Xray/val", image_height, batch_size)
    test_data = load_data("datasets/chest_Xray/test", image_height, batch_size)

    # Création du modèle
    model = create_model((image_height, image_width, 3))
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])

    # Entraînement du modèle
    model.fit(train_data, epochs=10, validation_data=validation_data)

    # Évaluation du modèle
    test_loss, test_acc = model.evaluate(test_data)
    print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")
    model.save("./save_model/my_model.h5")


if __name__ == "__main__":
    model_path = "save_model/my_model.h5"
    image_path = "datasets/chest_Xray/val/NORMAL/NORMAL2-IM-1438-0001.jpeg"
    # image_path = "datasets/chest_Xray/val/PNEUMONIA/person1951_bacteria_4882.jpeg"
    # train_model()
    predict_image(
        image_path,
        model_path,
    )
# evaluate_model("./save_model/my_models.h5", "datasets/chest_Xray/val", 150)
