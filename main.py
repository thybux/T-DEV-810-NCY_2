import pandas as pd

from src.data_handling import generate_labeled_data, create_dataset
from src.model import model_ia
from src.visualization import history_plot


def main():
    # labeliser les données
    data_dir = 'src/data/chest_Xray'

    df = generate_labeled_data(data_dir)

    output_csv_path = 'src/labeled/labeled_dataset.csv'
    df.to_csv(output_csv_path, index=False)

    # Préparer les données
    df = pd.read_csv('src/labeled/labeled_dataset.csv')

    train_df = df[df['type'] == 'train']
    val_df = df[df['type'] == 'val']
    test_df = df[df['type'] == 'test']

    train_dataset = create_dataset(train_df)
    val_dataset = create_dataset(val_df)
    test_dataset = create_dataset(test_df)

    # Entraîner le modèle
    model, history, val_loss, val_acc = model_ia(train_dataset, val_dataset, test_dataset)

    # Visualiser les résultats
    history_plot(history)


if __name__ == "__main__":
    main()
