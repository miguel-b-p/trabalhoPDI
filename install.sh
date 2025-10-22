#!/bin/bash
# Script de instalação do YOLO Benchmark System

echo "============================================"
echo "YOLO Benchmark System - Instalação"
echo "============================================"
echo ""

# Verifica Python
echo "Verificando Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 não encontrado. Instale Python 3.8 ou superior."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
echo "✅ Python $PYTHON_VERSION encontrado"
echo ""

# Cria ambiente virtual
echo "Criando ambiente virtual..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ Ambiente virtual criado"
else
    echo "✅ Ambiente virtual já existe"
fi
echo ""

# Ativa ambiente virtual
echo "Ativando ambiente virtual..."
source .venv/bin/activate
echo "✅ Ambiente virtual ativado"
echo ""

# Atualiza pip
echo "Atualizando pip..."
pip install --upgrade pip setuptools wheel
echo ""

# Instala dependências
echo "Instalando dependências..."
pip install -r requirements.txt
echo ""

# Verifica instalação
echo "Verificando instalação..."
python3 -c "
try:
    import ultralytics
    import torch
    import bokeh
    import ipywidgets
    import pandas
    print('✅ Todas as dependências principais instaladas')
except ImportError as e:
    print(f'❌ Erro na importação: {e}')
"
echo ""

# Cria diretórios necessários
echo "Criando diretórios..."
mkdir -p src/data
mkdir -p src/models
mkdir -p src/results/benchmarks
mkdir -p notebooks
echo "✅ Diretórios criados"
echo ""

# Copia .env.example se não existir .env
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Arquivo .env criado a partir de .env.example"
    fi
fi
echo ""

echo "============================================"
echo "✅ INSTALAÇÃO CONCLUÍDA!"
echo "============================================"
echo ""
echo "Para começar a usar:"
echo "  1. Ative o ambiente virtual: source .venv/bin/activate"
echo "  2. Execute Jupyter: jupyter lab"
echo "  3. Abra notebooks/01_quick_start.ipynb"
echo ""
echo "Ou execute o exemplo: python example_usage.py"
echo ""
