Project Name
==============================

This repo is a Starting Pack for DS projects. You can rearrange the structure to make it fits your project.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Training: PlantVillage species classifier (Keras)

This repository includes a training script to build an image classifier of plant species on the PlantVillage segmented dataset using Keras / TensorFlow.

### Data layout
- Root directory contains subfolders named `Species___Disease`, e.g. `Apple___Apple_scab`, `Cherry_(including_sour)___healthy`.
- The species label is the substring before `___`.
- Images are RGB 256×256.

### Script
`train_species_plantvillage_keras.py` will:
- Iterate all images under the dataset root and infer labels from folder names.
- Build a `tf.data` pipeline with light augmentations (horizontal flips, ±15° rotation, slight color/contrast jitter).
- Perform a stratified split train/val/test = 70/15/15 with seed=42.
- Train EfficientNetV2-B0 (ImageNet) with a softmax head and label smoothing (0.1).
- Use AdamW (fallback to Adam), EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint based on validation macro-F1.
- Save artifacts to `outputs/`:
  - `best_model.keras`
  - `class_index.json` (index→species mapping)
  - `history.csv` (training logs)
  - `training_curves.png`, `confusion_matrix.png`, `sanity_check.png`
  - `report.md` (summary with metrics and figures)

### Dependencies
Install with pip (minimal):
```
pip install -r requirements.txt
```
Notes:
- On Apple Silicon, you may prefer `tensorflow-macos` instead of `tensorflow`.
- GPU environments benefit from mixed precision (enabled automatically if a GPU is detected).

### Usage
Be careful to quote the dataset path if it contains spaces.
```
python train_species_plantvillage_keras.py \
  --data_root "/home/azureuser/dataset/plantvillage/data/plantvillage dataset/segmented" \
  --output_dir outputs \
  --epochs 100 \
  --batch_size 64
```

Optional arguments:
- `--img_size 256 256` (default)
- `--initial_lr 1e-3` (head), `--ft_lr 1e-4` (fine-tune)
- `--weight_decay 1e-4`, `--label_smoothing 0.1`
- `--fine_tune_at 50` (unfreeze top N layers in backbone)

Artifacts are saved in `outputs/`.
