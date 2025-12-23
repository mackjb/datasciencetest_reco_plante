# #!/usr/bin/env python3
# import subprocess
# from pathlib import Path

# # Liste des extensions à installer via WindSuf VSIX tool
# extensions = [
#     "ms-vscode.azure-account",
#     "ms-toolsai.vscode-ai",
#     "ms-toolsai.vscode-ai-remote",
#     "ms-vscode-remote.remote-ssh",
#     "ms-python.python",
#     "ms-toolsai.jupyter",
#     "ms-vscode-remote.remote-containers",
#     "ms-azuretools.vscode-docker"
# ]

# # Génère l'URL du marketplace pour chaque extension
# urls = [
#     f"https://marketplace.visualstudio.com/items?itemName={ext}"
#     for ext in extensions
# ]

# # Chemin vers le script VSIX tool (à adapter si besoin)
# VSIX_TOOL_PATH = Path(__file__).parent / "windsurf-vsix-tool" / "vsix-tool.py"

# for url in urls:
#     print(f"Installation de l'extension depuis {url}")
#     # Lancement du script en non-interactif : url, 'n' (pas de dépendances), 'y' (installer)
#     process = subprocess.run(
#         ["python3", str(VSIX_TOOL_PATH)],
#         input=f"{url}\nn\ny\n1\n",
#         text=True
#     )
#     if process.returncode != 0:
#         print(f"Erreur lors de l'installation de {url}")
#     else:
#         print(f"Succès pour {url}")
