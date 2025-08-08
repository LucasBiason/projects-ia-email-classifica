#!/bin/bash

echo "🔧 Corrigindo problema do SQLite..."

# Instalar dependências do sistema
echo "📦 Instalando dependências do sistema..."
sudo apt-get update
sudo apt-get install -y python3-dev libsqlite3-dev libffi-dev

# Reinstalar Python com suporte ao SQLite
echo "🐍 Reinstalando Python com suporte ao SQLite..."
pyenv install 3.12.11 --force

# Recriar ambiente Poetry
echo "📚 Recriando ambiente Poetry..."
cd /home/lucas-biason/Estudos/projects-ia-house-price-prediction
poetry env remove --all
poetry install

echo "✅ Problema do SQLite corrigido!"
echo "🎯 Agora você pode executar: make test" 