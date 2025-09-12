import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Données des modèles
models = ['XGBoost + PCA', 'XGBoost + LDA', 'XGBoost']
before_opt = [85.54, 85.92, 83.00]  # F1 scores avant optimisation
after_opt = [86.67, 81.61, 81.07]    # F1 scores après optimisation

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, before_opt, width, label='Avant Optimisation', color='#3498db')
rects2 = ax.bar(x + width/2, after_opt, width, label='Après Optimisation', color='#e74c3c')

# Ajout des étiquettes et du titre
ax.set_ylabel('Score F1 (%)', fontsize=12)
ax.set_title('Comparaison des Performances Avant/Après Optimisation', pad=20, fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=10)

# Ajout des valeurs sur les barres
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Enregistrement du graphique
output_path = '/workspaces/datasciencetest_reco_plante/figures/model_comparison_before_after.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Graphique de comparaison enregistré sous : {output_path}")
