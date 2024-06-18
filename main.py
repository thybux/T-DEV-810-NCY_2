import os

import pandas as pd

from src.data_handling import generate_labeled_data, create_dataset, increase_data
from src.model import model_ia
from src.visualization import history_plot


def main():
    # Labeliser les données
    data_dir = 'src/data/chest_Xray'
    output_csv_path = 'src/labeled/labeled_dataset.csv'
    save_dir_train = 'src/data/augmented/train'

    # Vérification de l'existence des données déjà augmentées
    if not os.path.exists(save_dir_train):
        # Si le répertoire d'augmentation n'existe pas, augmenter les données
        df = generate_labeled_data(data_dir)
        healthy_train_data = df.loc[(df['label'] == 'Healthy') & (df['type'] == 'train')]
        copy = 2
        increase_data(healthy_train_data, save_dir_train, copy)
        print('Train data increased and saved in:', save_dir_train)
    else:
        print('Train data already increased and exists in:', save_dir_train)

    # Vérification si le fichier CSV existe déjà
    already_exists = os.path.exists(output_csv_path)

    if already_exists:
        # Lire les données existantes
        df_existing = pd.read_csv(output_csv_path)
    else:
        df_existing = pd.DataFrame()

    # Générer les nouvelles données augmentées
    df_new = generate_labeled_data(save_dir_train, already_exists, is_increased=True)

    # Fusionner les nouvelles données avec les données existantes
    if not df_existing.empty:
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    # Sauvegarder les données combinées dans le fichier CSV
    df_combined.to_csv(output_csv_path, index=False)
    print(f'Updated dataset saved to: {output_csv_path}')

    # Préparer les données pour le modèle
    df = pd.read_csv(output_csv_path)

    label_map = {'Healthy': 0, 'Pneumonia': 1}

    train_df = df[df['type'] == 'train']
    val_df = df[df['type'] == 'val']
    test_df = df[df['type'] == 'test']

    train_dataset = create_dataset(train_df, label_map)
    val_dataset = create_dataset(val_df, label_map)
    test_dataset = create_dataset(test_df, label_map)

    # Entraîner le modèle
    model, history, val_loss, val_acc = model_ia(train_dataset, val_dataset)

    # Visualiser les résultats
    history_plot(history)


if __name__ == "__main__":
    main()
