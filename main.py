from src.data_handling import generate_labeled_data


def main():
    # Définir le chemin du répertoire du dataset
    data_dir = 'src/data/chest_Xray'  # Remplacez par le chemin de votre dataset

    # Générer le DataFrame labélisé
    df = generate_labeled_data(data_dir)

    # Sauvegarder le DataFrame dans un fichier CSV
    output_csv_path = 'src/labeled/labeled_dataset.csv'
    df.to_csv(output_csv_path, index=False)

    # Afficher un aperçu du DataFrame
    print(df.head())


if __name__ == "__main__":
    main()
