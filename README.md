# Projeto de Classificação de E-mails (Spam vs. Não Spam)

## Objetivo

Este projeto tem como objetivo construir um modelo de classificação de e-mails para detectar spam. O modelo é treinado usando um conjunto de dados de e-mails e disponibilizado através de uma API FastApi.

## Instalação

### Pré-requisitos

Certifique-se de ter os seguintes softwares instalados em sua máquina:

- [Python 3.8+](https://www.python.org/downloads/)
- [Docker](https://www.docker.com/get-started)

### Passo a Passo

1. **Clone o repositório:**

    ```bash
    git clone https://github.com/seu-usuario/projects-ia-email-classifica.git
    cd projects-ia-email-classifica
    ```

2. **Crie e ative um ambiente virtual:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # No Windows use `venv\Scripts\activate`
    ```

3. **Instale as dependências do projeto:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Execute a aplicação usando Docker:**

    ```bash
    make runapp-dev
    ```

5. **Acesse a API:**

    A API estará disponível em `http://localhost:5000`.

6. **Treine e teste o modelo:**

    Siga os exemplos de uso da API fornecidos na seção anterior para treinar e testar o modelo.
