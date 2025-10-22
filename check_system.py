#!/usr/bin/env python3
"""
Script de verificação do sistema
=================================

Verifica se todas as dependências e configurações estão corretas.
"""

import sys
from pathlib import Path


def check_python_version():
    """Verifica versão do Python."""
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ❌ Python 3.8+ necessário")
        return False
    else:
        print("  ✅ Versão OK")
        return True


def check_imports():
    """Verifica importações principais."""
    imports = {
        'ultralytics': 'YOLO',
        'torch': 'PyTorch',
        'bokeh': 'Bokeh',
        'ipywidgets': 'IPython Widgets',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'yaml': 'PyYAML',
        'PIL': 'Pillow',
    }
    
    all_ok = True
    for module, name in imports.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} não instalado")
            all_ok = False
    
    return all_ok


def check_cuda():
    """Verifica disponibilidade de CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA disponível")
            print(f"     GPU: {torch.cuda.get_device_name(0)}")
            print(f"     CUDA Version: {torch.version.cuda}")
        else:
            print("  ⚠️  CUDA não disponível (usando CPU)")
    except ImportError:
        print("  ❌ PyTorch não instalado")


def check_project_structure():
    """Verifica estrutura de diretórios."""
    required_dirs = [
        'src',
        'src/core',
        'src/interface',
        'src/results',
        'src/data',
        'src/models',
        'notebooks',
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ não encontrado")
            all_ok = False
    
    return all_ok


def check_core_modules():
    """Verifica módulos do core."""
    try:
        from src.core import Config, BenchmarkConfig, YOLOTrainer, BenchmarkRunner
        from src.results import BenchmarkVisualizer, ResultsAnalyzer
        from src.interface import YOLOBenchmarkInterface
        
        print("  ✅ Todos os módulos principais OK")
        return True
    except ImportError as e:
        print(f"  ❌ Erro ao importar módulos: {e}")
        return False


def check_config():
    """Verifica configuração."""
    try:
        from src.core import Config
        config = Config()
        
        print(f"  Project Root: {config.project_root}")
        print(f"  Data Path: {config.data_path}")
        print(f"  Models Path: {config.models_path}")
        print(f"  Results Path: {config.results_path}")
        
        # Verifica se diretórios foram criados
        if config.data_path.exists() and config.models_path.exists() and config.results_path.exists():
            print("  ✅ Configuração OK")
            return True
        else:
            print("  ⚠️  Alguns diretórios não existem (serão criados automaticamente)")
            return True
    except Exception as e:
        print(f"  ❌ Erro na configuração: {e}")
        return False


def main():
    """Função principal."""
    print("\n" + "="*80)
    print("YOLO BENCHMARK SYSTEM - VERIFICAÇÃO DO SISTEMA")
    print("="*80 + "\n")
    
    checks = {
        "Versão do Python": check_python_version,
        "Dependências": check_imports,
        "CUDA/GPU": check_cuda,
        "Estrutura do Projeto": check_project_structure,
        "Módulos Core": check_core_modules,
        "Configuração": check_config,
    }
    
    results = {}
    for name, check_func in checks.items():
        print(f"\n{name}:")
        print("-" * 80)
        results[name] = check_func()
    
    # Resumo
    print("\n" + "="*80)
    print("RESUMO")
    print("="*80 + "\n")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {name}")
    
    print(f"\nTotal: {passed}/{total} verificações passaram")
    
    if passed == total:
        print("\n🎉 Sistema pronto para uso!")
    else:
        print("\n⚠️  Alguns problemas foram encontrados. Resolva-os antes de continuar.")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
