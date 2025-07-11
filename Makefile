.PHONY: install flavia plantvillage new_plant_diseases_dataset plant_disease windsurf-extensions

SHELL := /bin/bash
.ONESHELL:

# Répertoire d'installation de Miniconda
CONDA_PREFIX := $(HOME)/miniconda
ENV_NAME     := conda_env

install:
	# 1) Configurer Git
	git config --global user.email "you@example.com"
	git config --global user.name  "Your Name"

	# 2) Mise à jour des paquets
	sudo apt-get update

	# 3) Installer Miniconda si nécessaire
	if [ ! -f "$(CONDA_PREFIX)/bin/conda" ]; then \
	  echo "⬇️  Téléchargement de Miniconda…"; \
	  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh; \
	  echo "⚙️  Installation silencieuse dans $(CONDA_PREFIX)…"; \
	  bash /tmp/miniconda.sh -b -p $(CONDA_PREFIX); \
	  rm /tmp/miniconda.sh; \
	else \
	  echo "⚠️  Miniconda déjà présent dans $(CONDA_PREFIX), skip installation."; \
	fi

	# 4) Initialiser Conda dans ~/.bashrc (pour bash)
	echo "🔧  Initialisation de Conda…"
	$(CONDA_PREFIX)/bin/conda init bash --no-user

	# 5) Créer ou mettre à jour l'environnement
	echo "🔍  Gestion de l'environnement '$(ENV_NAME)'…"
	if $(CONDA_PREFIX)/bin/conda env list | grep -qE "^$(ENV_NAME)[[:space:]]"; then \
	  echo "🔄  Mise à jour de '$(ENV_NAME)'…"; \
	  $(CONDA_PREFIX)/bin/conda env update -f conda_env.yml; \
	else \
	  echo "✨  Création de '$(ENV_NAME)'…"; \
	  $(CONDA_PREFIX)/bin/conda env create -f conda_env.yml; \
	fi

	# 6) Installer les extensions VSCode
	echo "🛠️  Installation des extensions VSCode…"; \
	code --install-extension ms-python.debugpy                            || true; \
	code --install-extension ms-python.python                             || true; \
	code --install-extension ms-toolsai.jupyter-keymap                    || true; \
	code --install-extension ms-toolsai.vscode-jupyter-slideshow         || true; \
	code --install-extension ms-toolsai.jupyter                           || true; \
	echo "✅  Extensions OK."

	# 7) Auto-activation à chaque nouveau shell
	#    Source le script conda.sh puis active l'env
	grep -qxF "source $(CONDA_PREFIX)/etc/profile.d/conda.sh" ~/.bashrc \
	  || echo "source $(CONDA_PREFIX)/etc/profile.d/conda.sh" >> ~/.bashrc
	grep -qxF "conda activate $(ENV_NAME)" ~/.bashrc \
	  || echo "conda activate $(ENV_NAME)" >> ~/.bashrc

	# 8) Remplacer ce shell par un Bash interactif
	echo "🚀  Nouvelle session Bash avec '$(ENV_NAME)' activé…"
	exec bash -i

# Cibles pour téléchargement des datasets
flavia:
	# Charger et activer l'environnement Conda
	source $(CONDA_PREFIX)/etc/profile.d/conda.sh
	conda activate $(ENV_NAME)
	# Télécharger le dataset Flavia
	python dataset/flavia/download_flavia.py

plantvillage:
	# Charger et activer l'environnement Conda
	source $(CONDA_PREFIX)/etc/profile.d/conda.sh
	conda activate $(ENV_NAME)
	# Télécharger le dataset PlantVillage
	python dataset/plantvillage/download_plantvillage.py

new_plant_diseases_dataset:
	# Charger et activer l'environnement Conda
	source $(CONDA_PREFIX)/etc/profile.d/conda.sh
	conda activate $(ENV_NAME)
	# Télécharger le dataset New Plant Diseases
	python dataset/new_plant_diseases_dataset/download_new_plant_diseases_dataset.py

plant_disease:
	# Charger et activer l'environnement Conda
	source $(CONDA_PREFIX)/etc/profile.d/conda.sh
	conda activate $(ENV_NAME)
	# Télécharger le dataset Plant Disease
	python dataset/plant_disease/download_plant_disease.py

.PHONY: windsurf-extensions
windsurf-extensions:
	# Cloner vsix-tool si nécessaire
	if [ ! -d ".devcontainer/windsurf-vsix-tool" ]; then \
		git clone https://github.com/twotreeszf/windsurf-vsix-tool.git .devcontainer/windsurf-vsix-tool; \
	else \
		echo "vsix-tool déjà cloné"; \
	fi
	# Installer les dépendances Python
	python3 -m pip install --no-cache-dir -r .devcontainer/windsurf-vsix-tool/requirements.txt
	echo "🛠️  Installation des extensions WindSuf VSCode…"
	python3 .devcontainer/install_windsurf_extensions.py