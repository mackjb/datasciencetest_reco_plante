from IPython.display import display
from src.helpers.helpers import PROJECT_ROOT
from src.data_loader.data_loader import load_plantvillage_all, load_plantvillage_five_images, generate_raw_data_plantvillage_segmented_all, generate_clean_data_plantvillage_segmented_all, generate_clean_and_resized, generate_segmented_clean_augmented_images, augment_minority_classes_cv2
import time
import hashlib
import pandas as pd
from pathlib import Path
from PIL import Image
# Feature importance pipeline imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt








if __name__ == "__main__":

    # Génération du CSV clean et des images 256x256 PNG en une seule méthode
    df_clean, output_dir = generate_clean_and_resized(force_refresh=False)
    print(df_clean.head())
    print(f"Clean CSV mis à jour et images PNG générées dans : {output_dir}")


    # Utilisation :
    df_aug = augment_minority_classes_cv2(df_clean, image_col='filepath', label_col='species')
    print(df_aug.head())

    #  --- Pipeline d'entraînement et explication SHAP ---
    # 1. Split train/val/test
    numeric_cols = ['file_size', 'width', 'height', 'num_channels', 'aspect_ratio']
    categorical_cols = ['extension', 'mode']
    X = df_clean[numeric_cols + categorical_cols]
    y = df_clean['species']

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    # 2. Pipeline : preproc + SMOTE + RF
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])
    pipeline = ImbPipeline([
        ('preproc', preprocessor),
        ('smote', SMOTE(sampling_strategy='minority', random_state=42)),
        ('clf', RandomForestClassifier(random_state=42)),
    ])

    # 3. Hyperparam tuning
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20],
    }
    grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print('Meilleurs paramètres :', grid.best_params_)

    # 4. Refit sur train+val
    best_pipe = grid.best_estimator_
    best_pipe.fit(X_train_val, y_train_val)

    # 5. Évaluation sur test
    y_pred = best_pipe.predict(X_test)
    print('Classification sur test :')
    print(classification_report(y_test, y_pred))

    # 6. SHAP
    # Background sample
    X_bg = preprocessor.transform(X_train_val.sample(100, random_state=42))
    explainer = shap.TreeExplainer(best_pipe.named_steps['clf'], X_bg)
    X_test_trans = preprocessor.transform(X_test)
    shap_values = explainer.shap_values(X_test_trans)

    # Global summary
    plt.figure()
    shap.summary_plot(shap_values, X_test_trans, feature_names=preprocessor.get_feature_names_out())
    plt.savefig('shap_summary.png', bbox_inches='tight')

    # Local force plots pour deux instances
    for i in [0, 1]:
        plt.figure()
        shap.force_plot(
            explainer.expected_value[1], shap_values[1][i], X_test_trans[i],
            feature_names=preprocessor.get_feature_names_out(), matplotlib=True
        )
        plt.savefig(f'force_plot_{i}.png', bbox_inches='tight')
    

    