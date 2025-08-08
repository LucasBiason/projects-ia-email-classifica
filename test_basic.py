#!/usr/bin/env python3
"""
Teste básico para verificar se o ambiente está funcionando.
"""

def test_basic_imports():
    """Testa imports básicos."""
    try:
        import fastapi
        print("✓ FastAPI importado com sucesso")
    except ImportError as e:
        print(f"✗ Erro ao importar FastAPI: {e}")
        return False
    
    try:
        import uvicorn
        print("✓ Uvicorn importado com sucesso")
    except ImportError as e:
        print(f"✗ Erro ao importar Uvicorn: {e}")
        return False
    
    try:
        import pydantic
        print("✓ Pydantic importado com sucesso")
    except ImportError as e:
        print(f"✗ Erro ao importar Pydantic: {e}")
        return False
    
    return True

def test_pytest():
    """Testa se o pytest está funcionando."""
    try:
        import pytest
        print("✓ Pytest importado com sucesso")
        return True
    except ImportError as e:
        print(f"✗ Erro ao importar Pytest: {e}")
        return False

if __name__ == "__main__":
    print("=== Teste de Ambiente ===")
    
    test1 = test_basic_imports()
    test2 = test_pytest()
    
    if test1 and test2:
        print("\n✅ Todos os testes básicos passaram!")
    else:
        print("\n❌ Alguns testes falharam!") 