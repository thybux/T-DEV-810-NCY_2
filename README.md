MNIST_Project/
│
├── data/                  # Dossier pour les données téléchargées ou générées
│   ├── raw/               # Données MNIST brutes
│   └── processed/         # Données MNIST transformées et prêtes à être utilisées
│
├── notebooks/             # Jupyter notebooks pour l'exploration de données et les tests
│   ├── data_exploration.ipynb
│   └── model_testing.ipynb
│
├── src/                   # Code source pour le projet
│   ├── __init__.py        # Rend ce répertoire un package Python
│   ├── data_handling.py   # Scripts pour charger et préparer les données
│   ├── model.py           # Définitions des modèles de machine learning
│   └── visualization.py   # Scripts pour la visualisation des données et des résultats
│
├── tests/                 # Tests unitaires
│   ├── __init__.py
│   ├── test_data_handling.py
│   └── test_model.py
│
├── outputs/               # Résultats générés, graphiques, fichiers de sortie, etc.
│   ├── figures/           # Graphiques et figures sauvegardés
│   └── models/            # Modèles entraînés sauvegardés
│
├── requirements.txt       # Dépendances Python nécessaires
└── README.md              # Documentation du projet, instructions, etc.

