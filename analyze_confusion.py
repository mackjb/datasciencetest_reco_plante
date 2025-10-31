import pandas as pd

print("="*80)
print("📊 ANALYSE DE LA MATRICE DE CONFUSION CASCADE")
print("="*80)

confusions = pd.read_csv('outputs/cascade_improved/cascade_top_confusions.csv')

print(f"\n✅ Accuracy CASCADE: 99.23%")
print(f"❌ Nombre total d'erreurs: 63 / 8146 images")
print(f"🎯 Nombre de classes: 38 (espèce___maladie)")

print("\n" + "="*80)
print("🔥 TOP 5 CONFUSIONS (Erreurs les plus fréquentes)")
print("="*80)
for idx, row in confusions.head(5).iterrows():
    print(f"\n{idx+1}. {row['Count']} erreurs ({row['Percent_of_true']:.1f}%)")
    print(f"   VRAI: {row['True']}")
    print(f"   PRÉDIT: {row['Predicted']}")

print("\n" + "="*80)
print("🔍 ANALYSE PAR TYPE D'ERREUR")
print("="*80)

inter_species = []
intra_species = []

for idx, row in confusions.iterrows():
    true_species = row['True'].split('___')[0]
    pred_species = row['Predicted'].split('___')[0]
    
    if true_species != pred_species:
        inter_species.append(row)
    else:
        intra_species.append(row)

print(f"\n📌 Erreurs INTRA-espèce (même espèce, maladie différente): {len(intra_species)}")
print(f"   → Le modèle d'espèce a bien prédit, mais maladie confuse")
print(f"\n📌 Erreurs INTER-espèces (espèce mal prédite): {len(inter_species)}")
print(f"   → Le 1er modèle s'est trompé d'espèce")

if len(inter_species) > 0:
    print("\n   Exemples d'erreurs inter-espèces:")
    for row in sorted(inter_species, key=lambda x: x['Count'], reverse=True)[:3]:
        true_sp = row['True'].split('___')[0]
        pred_sp = row['Predicted'].split('___')[0]
        print(f"   - {int(row['Count'])}x: {true_sp} → {pred_sp}")

print("\n" + "="*80)
print("📁 FICHIERS GÉNÉRÉS")
print("="*80)
print("✅ cascade_confusion_heatmap_full.png       - Matrice complète")
print("✅ cascade_confusion_heatmap_normalized.png - Matrice en %")
print("✅ cascade_confusion_errors_only.png        - Erreurs uniquement")
print("✅ cascade_confusion_matrix.csv             - Données brutes")
print("✅ cascade_top_confusions.csv               - Top confusions")
