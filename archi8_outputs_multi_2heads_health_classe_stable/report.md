# Architecture 8: Multi‑tâche (species, disease+healthy)
Date: 2025-11-05T01:16:29.296953

## Label spaces
- #species: 14
- #diseases+healthy: 21 (INCLUT healthy)

## Meilleures métriques (Validation/Test)
- Species: Val Acc=0.9988, Val F1=0.9986, Test Acc=0.9983, Test F1=0.9980
- Disease+Healthy (21 classes): Val Acc=0.9914, Val F1=0.9832, Test Acc=0.9919, Test F1=0.9847

## Classification Reports
- Species: classification_report_species.txt
- Disease: classification_report_disease_all.txt

## Figures
- Courbes: training_curves.png
- Confusion matrix (species): cm_species.png
- Confusion matrix (disease+healthy, 21 classes): cm_disease_all.png
- Sanity grid: sanity_check.png
