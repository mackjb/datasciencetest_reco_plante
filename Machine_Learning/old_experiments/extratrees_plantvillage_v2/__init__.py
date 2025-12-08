"""ExtraTrees PlantVillage v2 package.

Usage:
    python -m models.extratrees_plantvillage_v2

Ce package fournit une implémentation ExtraTrees avec pipeline robuste,
CV stratifiée, stratégies de gestion du déséquilibre et journalisation complète.
Métrique primaire: F1 macro (complément: balanced accuracy).
"""

from .extratrees_plantvillage_v2 import main, summarize_and_plot

__all__ = ["main", "summarize_and_plot"]
