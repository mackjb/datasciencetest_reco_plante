# Multi‑tâche (species, health, disease)
Date: 2025-10-31T11:35:37.744683

## Label spaces
- #species: 14
- #diseases (excl. healthy): 20

## Meilleures métriques (Validation/Test)
- Species: Val Acc=0.9955, Val F1=0.9946, Test Acc=0.9948, Test F1=0.9941
- Health:  Val Acc=0.9860,  Val F1=0.9826,  Test Acc=0.9874,  Test F1=0.9843
- Disease: Val Acc=0.9589, Val F1=0.9458, Test Acc=0.9561, Test F1=0.9432

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
