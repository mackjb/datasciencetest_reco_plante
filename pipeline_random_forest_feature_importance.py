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


def run_pipeline(test_size=0.2, random_state=42):
    # 1. Chargement et nettoyage
    df, _ = generate_clean_and_resized()
    df = df.dropna().reset_index(drop=True)
    X_paths = df['filepath'].tolist()
    y_labels = df['label'].values

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_paths, y_labels, test_size=test_size, random_state=random_state, stratify=y_labels
    )

    # 3. Définition de l’augmentation
    transforms = Compose([
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
        ShiftScaleRotate(p=0.5, shift_limit=0.1, scale_limit=0.1, rotate_limit=15),
    ])

    # 4. FeatureUnion pixels vs handcrafted
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

    # 5. Pipeline complet with sampler & classifier
    pipe = Pipeline([
        ('oversample', RandomOverSampler(random_state=random_state)),
        ('features', combined_features),
        ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=random_state)),
    ])

    # 6. Entraînement & évaluation
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return pipe


if __name__ == '__main__':
    run_pipeline()
