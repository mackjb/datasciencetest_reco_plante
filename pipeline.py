import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.data_loader.data_loader import generate_clean_and_resized


def load_and_prepare_data():
    """
    Charge et prépare les données via la génération clean et resized.
    Retourne un DataFrame sans valeurs manquantes.
    """
    df, _ = generate_clean_and_resized()
    df = df.dropna().reset_index(drop=True)
    return df


def balance_data(df, label_col='label', random_state=42):
    """
    Sur-échantillonne les classes minoritaires par duplication jusqu'à la taille maximale.
    """
    max_size = df[label_col].value_counts().max()
    df_list = [
        df[df[label_col] == cls].sample(max_size, replace=True, random_state=random_state)
        for cls in df[label_col].unique()
    ]
    df_bal = pd.concat(df_list).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df_bal


def prepare_features(df, label_col='label'):
    """
    Sépare features et labels, supprime colonnes non-numériques.
    """
    X = df.copy()
    # Colonnes à supprimer
    drop_cols = ['filepath', 'filename', 'extension', 'mode']
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    y = X[label_col]
    X = X.drop(columns=[label_col])
    return X, y


def run_pipeline(test_size=0.2, random_state=42):
    """
    Exécute la pipeline complète : chargement, balancing, standardisation, entraînement et évaluation.
    Affiche rapport de classification et importances.
    """
    # Chargement
    df = load_and_prepare_data()
    print(f"Loaded {len(df)} samples.")
    # Sur-échantillonnage
    df_bal = balance_data(df, random_state=random_state)
    print(f"Balanced dataset size: {len(df_bal)}")
    # Préparation des features
    X, y = prepare_features(df_bal)
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y)
    # Entraînement du modèle
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)
    # Évaluation
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    # Importance des features
    feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
    feat_importances = feat_importances.sort_values(ascending=False)
    print("Feature Importances:")
    print(feat_importances)
    return clf, scaler, feat_importances


if __name__ == '__main__':
    run_pipeline()
