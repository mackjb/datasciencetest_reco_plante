# Multi‑tâche (species, health, disease)
Date: 2025-10-31T20:07:23.668342

## Label spaces
- #species: 14
- #diseases (excl. healthy): 20

## Meilleures métriques (Validation/Test)
- Species: Val Acc=0.9950, Val F1=0.9942, Test Acc=0.9952, Test F1=0.9945
- Health:  Val Acc=0.9845,  Val F1=0.9806,  Test Acc=0.9865,  Test F1=0.9831
- Disease: Val Acc=0.9594, Val F1=0.9461, Test Acc=0.9582, Test F1=0.9461

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
