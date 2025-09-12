import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Données réelles de vos modèles (F1 scores en %)
models = ['XGBoost', 'XGBoost + LDA', 'XGBoost + PCA']
train_scores = [83.00, 85.92, 85.54]  # Avant optimisation
val_scores = [81.07, 81.61, 86.67]     # Après optimisation

# Création du graphique
plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")

# Largeur des barres
bar_width = 0.35
index = np.arange(len(models))

# Barres pour l'entraînement et la validation
bars1 = plt.bar(index - bar_width/2, train_scores, bar_width, 
               label='Entraînement (Base)', color='#3498db')
bars2 = plt.bar(index + bar_width/2, val_scores, bar_width, 
               label='Validation (Optimisé)', color='#e74c3c')

# Ajout des valeurs sur les barres
def add_values(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=10)

add_values(bars1)
add_values(bars2)

# Personnalisation du graphique
plt.title('Comparaison des Performances Réelles des Modèles', pad=20, fontsize=16, fontweight='bold')
plt.xlabel('Modèles', labelpad=15, fontsize=12)
plt.ylabel('Score F1 (%)', labelpad=15, fontsize=12)
plt.xticks(index, models, fontsize=11)
plt.ylim(0, 100)
plt.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)

# Ajout d'annotations
effects = [
    "Meilleure performance en validation",
    "Bonne stabilité",
    "Légère baisse en validation"
]

for i, effect in enumerate(effects):
    plt.annotate(effect, 
                xy=(i, val_scores[i]), 
                xytext=(i, val_scores[i] + 5),
                ha='center', va='bottom',
                fontsize=10,
                arrowprops=dict(arrowstyle="->", color='black'))

plt.tight_layout()

# Enregistrement du graphique
output_path = '/workspaces/datasciencetest_reco_plante/figures/actual_models_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Graphique des modèles réels enregistré sous : {output_path}")
