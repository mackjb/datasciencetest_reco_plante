# Multi‑tâche (species, health, disease)
Date: 2025-10-29T17:54:59.107385

## Label spaces
- #species: 14
- #diseases (excl. healthy): 20

## Meilleures métriques (Validation/Test)
- Species: Val Acc=0.9956, Val F1=0.9948, Test Acc=0.9942, Test F1=0.9934
- Health:  Val Acc=0.9848,  Val F1=0.9809,  Test Acc=0.9872,  Test F1=0.9840
- Disease: Val Acc=0.9582, Val F1=0.9434, Test Acc=0.9577, Test F1=0.9438

## Classification Reports
- Species: `reports/classification_report_species.txt` | `reports/classification_report_species.json`
- Health: `reports/classification_report_health.txt` | `reports/classification_report_health.json`
- Disease: `reports/classification_report_disease.txt` | `reports/classification_report_disease.json`

## Figures
- Courbes: training_curves.png
- Confusion matrix (species): cm_species.png
- Confusion matrix (health): cm_health.png
- Confusion matrix (disease): cm_disease.png
- Sanity grid: sanity_check.png
