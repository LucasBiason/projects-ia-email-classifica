#!/bin/bash

echo "🔧 Corrigindo ambiente Poetry..."

# Recriar ambiente Poetry
echo "📚 Recriando ambiente Poetry..."
poetry env remove --all
poetry install

echo "✅ Ambiente corrigido!"
echo "🎯 Agora você pode executar: make test" 