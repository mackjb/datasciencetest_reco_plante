import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Données simulées pour montrer le surapprentissage
np.random.seed(42)
train_sizes = np.linspace(0.1, 1.0, 10)

# Modèle avec surapprentissage modéré
def generate_overfitting_curves(base_score, overfit_factor=0.15):
    train = base_score + np.random.normal(0, 0.01, len(train_sizes))
    val = base_score - overfit_factor + np.random.normal(0, 0.01, len(train_sizes))
    return np.clip(train, 0, 1), np.clip(val, 0, 1)

# Génération des courbes
data = {
    'Surapprentissage Faible': {
        'train_sizes': train_sizes,
        'train_scores': generate_overfitting_curves(0.85, 0.05)[0],
        'val_scores': generate_overfitting_curves(0.85, 0.05)[1]
    },
    'Surapprentissage Fort': {
        'train_sizes': train_sizes,
        'train_scores': generate_overfitting_curves(0.95, 0.25)[0],
        'val_scores': generate_overfitting_curves(0.95, 0.25)[1]
    },
    'Sous-apprentissage': {
        'train_sizes': train_sizes,
        'train_scores': generate_overfitting_curves(0.65, 0.02)[0],
        'val_scores': generate_overfitting_curves(0.65, 0.02)[1]
    }
}

# Création du graphique
plt.figure(figsize=(16, 10))

# Style pour meilleure lisibilité
sns.set_style("whitegrid")
colors = ['#3498db', '#e74c3c', '#2ecc71']

for idx, (model_name, model_data) in enumerate(data.items()):
    color = colors[idx % len(colors)]
    plt.plot(model_data['train_sizes']*100, model_data['train_scores']*100, 
             'o-', color=color, linewidth=2, markersize=8, 
             label=f'{model_name} (Entraînement)')
    plt.plot(model_data['train_sizes']*100, model_data['val_scores']*100, 
             'o--', color=color, linewidth=2, markersize=8,
             label=f'{model_name} (Validation)')
    
    # Ajout de la zone d'écart
    plt.fill_between(model_data['train_sizes']*100, 
                    model_data['val_scores']*100,
                    model_data['train_scores']*100,
                    color=color, alpha=0.1)

# Personnalisation du graphique
plt.title('Analyse du Surapprentissage des Modèles', pad=20, fontsize=16, fontweight='bold')
plt.xlabel('Pourcentage des Données d\'Entraînement (%)', labelpad=15, fontsize=12)
plt.ylabel('Score F1 (%)', labelpad=15, fontsize=12)
plt.xticks(np.linspace(10, 100, 10), fontsize=10)
plt.yticks(np.linspace(0, 100, 11), fontsize=10)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)

# Ajout d'annotations explicatives
plt.annotate('Écart important = Surapprentissage', 
             xy=(80, 90), xytext=(50, 95),
             arrowprops=dict(facecolor='black', shrink=0.05),
             ha='center', fontsize=11, fontweight='bold')

plt.annotate('Écart faible = Bonne généralisation', 
             xy=(80, 60), xytext=(50, 65),
             arrowprops=dict(facecolor='black', shrink=0.05),
             ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()

# Enregistrement du graphique
output_path = '/workspaces/datasciencetest_reco_plante/figures/learning_curves_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Graphique des courbes d'apprentissage enregistré sous : {output_path}")
