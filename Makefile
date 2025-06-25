.PHONY: install

SHELL := /bin/bash
.ONESHELL:

# Répertoire d'installation de Miniconda
CONDA_PREFIX := $(HOME)/miniconda
ENV_NAME := conda_env



install:
	# Configurer Git globalement
	git config --global user.email "you@example.com"
	git config --global user.name "Your Name"

	sudo apt-get update
	# 1) Installation de Miniconda si nécessaire
	if [ -d "$(CONDA_PREFIX)" ]; then
		echo "⚠️ Miniconda est déjà installé dans $(CONDA_PREFIX), installation annulée."
	else
		echo "1) Télécharger le script d'installation de Miniconda..."
		wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
		echo "2) Installer en mode silencieux dans $(CONDA_PREFIX)..."
		bash /tmp/miniconda.sh -b -p $(CONDA_PREFIX)
		rm /tmp/miniconda.sh
	fi

	# 2) Initialiser Conda pour cette session et les suivantes
	eval "$(CONDA_PREFIX)/bin/conda shell.bash hook"
	$(CONDA_PREFIX)/bin/conda init --no-user

	# 3) Sourcing pour appliquer immédiatement
	source ~/.bashrc

	# 4) Gérer l'environnement Conda en spécifiant le binaire complet
	echo "Vérification de l'environnement Conda '$(ENV_NAME)'…"
	if $(CONDA_PREFIX)/bin/conda env list | grep -qE "^$(ENV_NAME)[[:space:]]"; then
		echo "Mise à jour de l'environnement '$(ENV_NAME)'"
		$(CONDA_PREFIX)/bin/conda env update -f conda_env.yml
	else
		echo "Création de l'environnement '$(ENV_NAME)'"
		$(CONDA_PREFIX)/bin/conda env create -f conda_env.yml
	fi

	# 5) Installer les extensions VSCode
	echo "Installation des extensions VSCode..."
	code --install-extension ms-python.debugpy || true
	code --install-extension ms-python.python || true
	code --install-extension ms-toolsai.jupyter-keymap || true
	code --install-extension ms-toolsai.vscode-jupyter-slideshow || true
	code --install-extension ms-toolsai.jupyter || true
	echo "✅ Extensions VSCode installées."


# (base) ubuntu@ip-172-31-24-93:~/datasciencetest_reco_plante$ git config --global user.email "you@example.com"
# (base) ubuntu@ip-172-31-24-93:~/datasciencetest_reco_plante$ git config --global user.name "Your Name"

# .PHONY: install

# install:
# 	sudo apt-get update

# 	@echo "1) Télécharger le script d'installation de Miniconda..."
# 	@wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
# 	@echo "2) Lancer l'installation en mode silencieux dans $(CONDA_PREFIX)..."
# 	@bash /tmp/miniconda.sh -b -p $(CONDA_PREFIX)
# 	@rm /tmp/miniconda.sh
# 	@echo "3) Initialiser Conda dans votre shell (bash)..."
# 	@eval "$$($(CONDA_PREFIX)/bin/conda shell.bash hook)"
# 	@$(CONDA_PREFIX)/bin/conda init
# 	@echo "Installation terminée ! Veuillez redémarrer votre terminal ou exécuter 'source ~/.bashrc'."


# 	sudo apt-get install -y libgl1-mesa-glx
