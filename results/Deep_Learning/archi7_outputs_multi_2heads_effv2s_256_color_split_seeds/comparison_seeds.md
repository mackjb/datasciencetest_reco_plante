# Comparaison Multi-tâche 2 têtes: 5 runs avec seeds différents

## Résultats individuels

       dir   seed  test_acc_species  test_f1_species  val_acc_species  val_f1_species  test_acc_disease  test_f1_disease  val_acc_disease  val_f1_disease  test_macro_precision_species  test_macro_recall_species  test_macro_f1_species_detailed  test_macro_precision_disease  test_macro_recall_disease  test_macro_f1_disease_detailed
seed101112 101112            0.9989           0.9988           0.9995          0.9995            0.9944           0.9899           0.9935          0.9890                        0.9993                     0.9983                          0.9988                        0.9895                     0.9908                          0.9899
   seed123    123            0.9985           0.9983           0.9999          0.9999            0.9946           0.9914           0.9937          0.9886                        0.9984                     0.9982                          0.9983                        0.9911                     0.9917                          0.9914
    seed42     42            0.9990           0.9989           0.9998          0.9998            0.9932           0.9881           0.9932          0.9885                        0.9992                     0.9985                          0.9989                        0.9871                     0.9904                          0.9881
   seed456    456            0.9993           0.9992           0.9999          0.9999            0.9963           0.9942           0.9952          0.9915                        0.9995                     0.9989                          0.9992                        0.9941                     0.9943                          0.9942
   seed789    789            0.9990           0.9991           0.9998          0.9998            0.9935           0.9895           0.9920          0.9857                        0.9994                     0.9987                          0.9991                        0.9885                     0.9911                          0.9895

## Statistiques (Test Set)

### Tâche: SPECIES

#### Test Acc
- Mean: 0.9989
- Std: 0.0003
- Min: 0.9985
- Max: 0.9993
- Range: 0.0008
- **CV: 0.03%**
  - ✅ **Très stable** (CV < 1%)

#### Test F1
- Mean: 0.9989
- Std: 0.0004
- Min: 0.9983
- Max: 0.9992
- Range: 0.0009
- **CV: 0.04%**
  - ✅ **Très stable** (CV < 1%)

### Tâche: DISEASE

#### Test Acc
- Mean: 0.9944
- Std: 0.0012
- Min: 0.9932
- Max: 0.9963
- Range: 0.0031
- **CV: 0.12%**
  - ✅ **Très stable** (CV < 1%)

#### Test F1
- Mean: 0.9906
- Std: 0.0023
- Min: 0.9881
- Max: 0.9942
- Range: 0.0061
- **CV: 0.23%**
  - ✅ **Très stable** (CV < 1%)

## Interprétation globale

- **SPECIES (Test F1)**: Mean=0.9989, Std=0.0004, CV=0.04%
- **DISEASE (Test F1)**: Mean=0.9906, Std=0.0023, CV=0.23%

## Recommandation

- Si CV < 3% pour les deux tâches : Résultats robustes, un seul run suffit.
- Si 3% <= CV < 5% : Rapporter la moyenne ± écart-type.
- Si CV >= 5% : Investiguer les causes (instabilité).
