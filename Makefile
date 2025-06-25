.PHONY: install

SHELL := /bin/bash
.ONESHELL:

# Répertoire d'installation de Miniconda
CONDA_PREFIX := $(HOME)/miniconda
ENV_NAME     := conda_env

install:
	# Configurer Git globalement
	git config --global user.email "you@example.com"
	git config --global user.name  "Your Name"

	sudo apt-get update

	# 1) Vérifier si 'conda' existe déjà
	if command -v conda &> /dev/null; then \
		echo "⚠️  La commande 'conda' existe déjà : $$(conda --version). Installation de Miniconda annulée."; \
	else \
		# 2) Installer Miniconda si nécessaire
		if [ -d "$(CONDA_PREFIX)" ]; then \
			echo "⚠️  Miniconda est déjà présent dans $(CONDA_PREFIX), installation annulée."; \
		else \
			echo "1) Téléchargement du script Miniconda…"; \
			wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh; \
			echo "2) Installation silencieuse dans $(CONDA_PREFIX)…"; \
			bash /tmp/miniconda.sh -b -p $(CONDA_PREFIX); \
			rm /tmp/miniconda.sh; \
		fi; \
	fi

	# 3) Initialiser Conda pour cette session
	eval "$$($(CONDA_PREFIX)/bin/conda shell.bash hook)"
	$(CONDA_PREFIX)/bin/conda init --no-user

	# 4) Création ou mise à jour de l'env
	echo "Vérification de l'env '$(ENV_NAME)'…"
	if $(CONDA_PREFIX)/bin/conda env list | grep -qE "^$(ENV_NAME)[[:space:]]"; then \
		echo "🔄 Mise à jour de '$(ENV_NAME)'…"; \
		$(CONDA_PREFIX)/bin/conda env update -f conda_env.yml; \
	else \
		echo "✨ Création de '$(ENV_NAME)'…"; \
		$(CONDA_PREFIX)/bin/conda env create -f conda_env.yml; \
	fi

	# 5) Extensions VSCode
	echo "Installation des extensions VSCode…"
	code --install-extension ms-python.debugpy                            || true
	code --install-extension ms-python.python                             || true
	code --install-extension ms-toolsai.jupyter-keymap                    || true
	code --install-extension ms-toolsai.vscode-jupyter-slideshow         || true
	code --install-extension ms-toolsai.jupyter                           || true
	echo "✅ Extensions VSCode OK."

	# 6) Auto-activation de l'env sur chaque nouveau shell
	grep -qxF "conda activate $(ENV_NAME)" ~/.bashrc || \
	  echo "conda activate $(ENV_NAME)" >> ~/.bashrc

	# 7) Lancer un nouveau shell Bash qui lira ~/.bashrc et activera l'env
	echo "🚀 Démarrage d’un nouveau Bash interactif avec '$(ENV_NAME)' activé…"
	exec bash -i