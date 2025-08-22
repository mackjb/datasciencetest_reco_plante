"""Module d'entrée pour exécuter le pipeline ExtraTrees v2.

Usage:
    python -m models.extratrees_plantvillage_v2
"""
from .extratrees_plantvillage_v2 import main


def _run():
    # Point d'entrée isolé (plus facile à tester/mocker si besoin)
    main()


if __name__ == "__main__":
    _run()
