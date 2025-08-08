#!/usr/bin/env python3
"""
Script para treinar o modelo de classificação de emails.
"""

from app.email_classifier import EmailClassifier

def main():
    """Treina o modelo de classificação de emails."""
    print("Iniciando treinamento do modelo...")
    
    try:
        classifier = EmailClassifier()
        classifier.train()
        print("Modelo treinado e salvo com sucesso!")
    except Exception as e:
        print(f"Erro ao treinar o modelo: {e}")
        raise

if __name__ == "__main__":
    main() 