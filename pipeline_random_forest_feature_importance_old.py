import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion, Pipeline as SkPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from albumentations import Compose, HorizontalFlip, RandomRotate90, ShiftScaleRotate
from PIL import Image

from src.data_loader.data_loader import generate_clean_and_resized
from src.helpers.helpers import (
    compute_hu_features,
    compute_fourier_energy,
    compute_hog_features,
    compute_pixel_ratio_and_segments,
)

class ImageLoader(BaseEstimator, TransformerMixin):
    def __init__(self, target_size=(256, 256), transforms=None):
        self.target_size = target_size
        self.transforms = transforms

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        imgs = []
        for p in X:
            img = imread(p)
            img = resize(img, self.target_size, preserve_range=True, anti_aliasing=True)
            if self.transforms:
                img = self.transforms(image=img)['image']
            img = img / 255.0
            imgs.append(img.ravel())
        return np.vstack(imgs)

class HandCraftedFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for p in X:
            img = Image.open(p).convert("L")
            arr = np.array(img)
            hu = compute_hu_features(arr)
            fourier = compute_fourier_energy(arr)
            hog = compute_hog_features(arr)
            pix = compute_pixel_ratio_and_segments(arr)
            feats.append([
                hu['phi1_distingue_large_vs_etroit'],
                hu['phi2_distinction_elongation_forme'],
                hu['phi3_asymetrie_maladie'],
                hu['phi4_symetrie_diagonale_forme'],
                hu['phi5_concavite_extremites'],
                hu['phi6_decalage_torsion_maladie'],
                hu['phi7_asymetrie_complexe'],
                fourier['energie_basse_forme_feuille'],
                fourier['energie_moyenne_texture_veines'],
                fourier['energie_haute_details_maladie'],
                hog['hog_moyenne_contours_forme'],
                hog['hog_ecarttype_texture'],
                pix['pixel_ratio'],
                pix['leaf_segments'],
            ])
        return np.array(feats)

if __name__ == '__main__':
    # 1. Chargement
    df, _ = generate_clean_and_resized()
    df = df.dropna().reset_index(drop=True)
    X_paths = df['filepath'].tolist()
    y_labels = df['label'].values

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_paths, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )

    # 3. Augmentation
    transforms = Compose([
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
        ShiftScaleRotate(p=0.5, shift_limit=0.1, scale_limit=0.1, rotate_limit=15),
    ])

    # 4. FeatureUnion
    combined_features = FeatureUnion([
        ('pixels', SkPipeline([
            ('loader', ImageLoader(target_size=(256,256), transforms=transforms)),
            ('scale_px', StandardScaler()),
        ])),
        ('handmade', SkPipeline([
            ('features', HandCraftedFeatures()),
            ('scale_hc', StandardScaler()),
        ])),
    ])

    # 5. Pipeline complet
    pipeline = Pipeline([
        ('features', combined_features),
        ('oversample', RandomOverSampler(random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)),
    ])

    # 6. Entraînement
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

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


def augment_fn(X, y, transforms=None, random_state=42):
    """
    Augmente les classes minoritaires par transformations d'images.
    """
    df = X.copy().reset_index(drop=True)
    labels = pd.Series(y).reset_index(drop=True)
    counts = labels.value_counts()
    max_count = counts.max()
    new_rows = []
    new_labels = []
    for cls, cnt in counts.items():
        n_to_gen = max_count - cnt
        if n_to_gen <= 0:
            continue
        cls_df = df[labels == cls]
        for i in range(n_to_gen):
            sample = cls_df.sample(1, random_state=random_state).iloc[0]
            img_arr = np.array(Image.open(sample['filepath']))
            augmented = transforms(image=img_arr)['image']
            # calcul des features
            gray = np.array(Image.fromarray(augmented).convert('L'))
            hu = compute_hu_features(gray)
            fourier = compute_fourier_energy(gray)
            hog = compute_hog_features(gray)
            pix = compute_pixel_ratio_and_segments(gray)
            new = sample.to_dict()
            new.update({
                'phi1_distingue_large_vs_etroit': hu['phi1_distingue_large_vs_etroit'],
                'phi2_distinction_elongation_forme': hu['phi2_distinction_elongation_forme'],
                'phi3_asymetrie_maladie': hu['phi3_asymetrie_maladie'],
                'phi4_symetrie_diagonale_forme': hu['phi4_symetrie_diagonale_forme'],
                'phi5_concavite_extremites': hu['phi5_concavite_extremites'],
                'phi6_decalage_torsion_maladie': hu['phi6_decalage_torsion_maladie'],
                'phi7_asymetrie_complexe': hu['phi7_asymetrie_complexe'],
                'energie_basse_forme_feuille': fourier['energie_basse_forme_feuille'],
                'energie_moyenne_texture_veines': fourier['energie_moyenne_texture_veines'],
                'energie_haute_details_maladie': fourier['energie_haute_details_maladie'],
                'hog_moyenne_contours_forme': hog['hog_moyenne_contours_forme'],
                'hog_ecarttype_texture': hog['hog_ecarttype_texture'],
                'pixel_ratio': pix['pixel_ratio'],
                'leaf_segments': pix['leaf_segments']
            })
            new_rows.append(new)
            new_labels.append(cls)
    if new_rows:
        df_aug = pd.DataFrame(new_rows)
        labels_aug = pd.Series(new_labels)
        X_res = pd.concat([df, df_aug], ignore_index=True)
        y_res = pd.concat([labels, labels_aug], ignore_index=True)
        return X_res, y_res
    return X, y


def run_pipeline(test_size=0.2, random_state=42):
    """
    Exécute la pipeline complète : chargement, augmentation ciblée des classes minoritaires,
    standardisation, entraînement et évaluation via imblearn.pipeline.Pipeline.
    """
    # Chargement et préparation initiale
    df = load_and_prepare_data()
    df = df.dropna().reset_index(drop=True)
    # Séparation train/test sur DataFrame brut
    y = df['label']
    X_df = df
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=random_state, stratify=y)
    # Définition des transformations d'augmentation
    transforms = Compose([
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5)
    ])
    # Construction de la pipeline
    pipeline = ImbPipeline([
        ('augment', FunctionSampler(func=augment_fn, kw_args={'transforms': transforms, 'random_state': random_state})),
        ('feat', FunctionTransformer(lambda df: prepare_features(df)[0], validate=False)),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=random_state))
    ])
    # Entraînement
    pipeline.fit(X_train_df, y_train)
    # Prédiction
    y_pred = pipeline.predict(X_test_df)
    # Évaluation
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    # Importance des features
    clf = pipeline.named_steps['clf']
    feature_names = prepare_features(X_train_df)[0].columns
    feat_importances = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("Feature Importances:")
    print(feat_importances)
    return pipeline
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
