# Script PowerShell pour gérer le dev container GPU
# Usage: .\dev-commands.ps1 [commande]

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ContainerName = "gpu-dev-env"
$ImageName = "gpu-dev-env:latest"

function Show-Help {
    Write-Host ""
    Write-Host "🚀 Dev Container GPU - Commandes disponibles" -ForegroundColor Cyan
    Write-Host "=============================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  build       " -ForegroundColor Green -NoNewline
    Write-Host "- Construire l'image Docker"
    Write-Host "  run         " -ForegroundColor Green -NoNewline
    Write-Host "- Lancer le container"
    Write-Host "  stop        " -ForegroundColor Green -NoNewline
    Write-Host "- Arrêter le container"
    Write-Host "  restart     " -ForegroundColor Green -NoNewline
    Write-Host "- Redémarrer le container"
    Write-Host "  shell       " -ForegroundColor Green -NoNewline
    Write-Host "- Ouvrir un shell dans le container"
    Write-Host "  logs        " -ForegroundColor Green -NoNewline
    Write-Host "- Voir les logs du container"
    Write-Host "  test-gpu    " -ForegroundColor Green -NoNewline
    Write-Host "- Tester le GPU dans le container"
    Write-Host "  jupyter     " -ForegroundColor Green -NoNewline
    Write-Host "- Lancer Jupyter Lab"
    Write-Host "  check-gpu   " -ForegroundColor Green -NoNewline
    Write-Host "- Vérifier le GPU sur l'hôte"
    Write-Host "  clean       " -ForegroundColor Green -NoNewline
    Write-Host "- Nettoyer les containers/images"
    Write-Host "  rebuild     " -ForegroundColor Green -NoNewline
    Write-Host "- Rebuild complet (no cache)"
    Write-Host ""
    Write-Host "Exemples:" -ForegroundColor Yellow
    Write-Host "  .\dev-commands.ps1 build"
    Write-Host "  .\dev-commands.ps1 run"
    Write-Host "  .\dev-commands.ps1 test-gpu"
    Write-Host ""
}

function Build-Image {
    Write-Host "🔨 Construction de l'image Docker..." -ForegroundColor Cyan
    docker build -t $ImageName "$PSScriptRoot"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Image construite avec succès!" -ForegroundColor Green
    } else {
        Write-Host "❌ Erreur lors de la construction" -ForegroundColor Red
    }
}

function Run-Container {
    Write-Host "🚀 Démarrage du container..." -ForegroundColor Cyan
    
    # Vérifier si le container existe déjà
    $existing = docker ps -a --filter "name=$ContainerName" --format "{{.Names}}"
    if ($existing) {
        Write-Host "⚠️  Container existant trouvé. Démarrage..." -ForegroundColor Yellow
        docker start $ContainerName
    } else {
        docker run -it --name $ContainerName `
            --gpus all `
            --shm-size=4g `
            --ipc=host `
            -v "${ProjectRoot}:/workspace" `
            -v "$env:USERPROFILE\.nv:/root/.nv" `
            -v "$env:USERPROFILE\.cache:/root/.cache" `
            -p 8888:8888 `
            -p 6006:6006 `
            -e NVIDIA_VISIBLE_DEVICES=all `
            -e NVIDIA_DRIVER_CAPABILITIES=compute,utility `
            -e TF_FORCE_GPU_ALLOW_GROWTH=true `
            $ImageName
    }
}

function Stop-Container {
    Write-Host "🛑 Arrêt du container..." -ForegroundColor Cyan
    docker stop $ContainerName
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Container arrêté" -ForegroundColor Green
    }
}

function Restart-Container {
    Write-Host "🔄 Redémarrage du container..." -ForegroundColor Cyan
    docker restart $ContainerName
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Container redémarré" -ForegroundColor Green
    }
}

function Open-Shell {
    Write-Host "💻 Ouverture du shell..." -ForegroundColor Cyan
    docker exec -it $ContainerName /bin/bash
}

function Show-Logs {
    Write-Host "📜 Logs du container:" -ForegroundColor Cyan
    docker logs $ContainerName
}

function Test-GPU {
    Write-Host "🧪 Test GPU dans le container..." -ForegroundColor Cyan
    docker exec -it $ContainerName python /workspace/.devcontainer/test_gpu.py
}

function Start-Jupyter {
    Write-Host "📓 Démarrage de Jupyter Lab..." -ForegroundColor Cyan
    Write-Host "Accès: http://localhost:8888" -ForegroundColor Yellow
    docker exec -it $ContainerName jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
}

function Check-HostGPU {
    Write-Host "🎮 Vérification GPU sur l'hôte..." -ForegroundColor Cyan
    nvidia-smi
    
    Write-Host ""
    Write-Host "🐳 Test Docker GPU..." -ForegroundColor Cyan
    docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
}

function Clean-Docker {
    Write-Host "🧹 Nettoyage Docker..." -ForegroundColor Cyan
    
    Write-Host "Suppression des containers arrêtés..."
    docker container prune -f
    
    Write-Host "Suppression des images non utilisées..."
    docker image prune -f
    
    Write-Host "✅ Nettoyage terminé" -ForegroundColor Green
}

function Rebuild-Image {
    Write-Host "🔨 Rebuild complet (no cache)..." -ForegroundColor Cyan
    docker build --no-cache -t $ImageName "$PSScriptRoot"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Image reconstruite avec succès!" -ForegroundColor Green
    } else {
        Write-Host "❌ Erreur lors de la reconstruction" -ForegroundColor Red
    }
}

# Router les commandes
switch ($Command.ToLower()) {
    "build" { Build-Image }
    "run" { Run-Container }
    "stop" { Stop-Container }
    "restart" { Restart-Container }
    "shell" { Open-Shell }
    "logs" { Show-Logs }
    "test-gpu" { Test-GPU }
    "jupyter" { Start-Jupyter }
    "check-gpu" { Check-HostGPU }
    "clean" { Clean-Docker }
    "rebuild" { Rebuild-Image }
    "help" { Show-Help }
    default {
        Write-Host "❌ Commande inconnue: $Command" -ForegroundColor Red
        Show-Help
    }
}
