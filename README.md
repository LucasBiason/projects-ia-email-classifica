# 📧 Email Classification Service

Um serviço de classificação de emails usando Machine Learning para identificar spam e emails legítimos (ham).

## 🚀 Funcionalidades

- **Classificação Automática**: Identifica spam vs emails legítimos
- **API RESTful**: Endpoints para classificação em tempo real
- **Modelo ML**: Multinomial Naive Bayes com CountVectorizer
- **Validação de Dados**: Schemas Pydantic para validação
- **Testes Completos**: 100% de cobertura de código
- **Docker**: Containerização para fácil deploy

## 🛠️ Tecnologias

- **Framework**: FastAPI
- **ML**: scikit-learn (MultinomialNB, CountVectorizer)
- **Validação**: Pydantic
- **Testes**: pytest
- **Containerização**: Docker
- **Python**: 3.11+

## 📋 Requisitos

- Python 3.11+
- Docker (opcional)
- Dados de treinamento em `data/emails.csv`

## 🚀 Instalação

### Local

```bash
# Clone o repositório
git clone <repository-url>
cd projects-ia-email-classifica

# Instale as dependências
pip install -r requirements.txt

# Execute a aplicação
uvicorn app.main:app --reload
```

### Docker

```bash
# Build e execute com Docker Compose
docker compose up --build

# Ou apenas o container de teste
docker compose run --rm test
```

## 📖 Uso

### API Endpoints

#### 1. Status do Serviço
```bash
GET /
```
Resposta:
```json
{
  "message": "Email Classification Service is online!",
  "status": "healthy",
  "classifier_ready": true
}
```

#### 2. Health Check
```bash
GET /health
```
Resposta:
```json
{
  "status": "healthy",
  "classifier_ready": true
}
```

#### 3. Classificação de Email
```bash
POST /predict
Content-Type: application/json

{
  "message": "Win a free iPhone now! Click here!"
}
```
Resposta:
```json
{
  "prediction": "spam"
}
```

### Exemplos de Uso

```python
import requests

# Classificar email
response = requests.post(
    "http://localhost:8000/predict",
    json={"message": "Hello, how are you?"}
)
result = response.json()
print(f"Classificação: {result['prediction']}")
```

## 🧪 Testes

### Executar Testes
```bash
# Testes locais
pytest

# Testes com cobertura
pytest --cov=app --cov-report=term-missing

# Testes no Docker
docker compose run --rm test
```

### Cobertura de Código
```bash
# Gerar relatório de cobertura
pytest --cov=app --cov-report=html
```

## 📁 Estrutura do Projeto

```
projects-ia-email-classifica/
├── app/
│   ├── __init__.py
│   ├── main.py              # Aplicação FastAPI
│   ├── email_classifier.py  # Modelo ML
│   ├── schemas.py           # Schemas Pydantic
│   └── views.py             # Endpoints da API
├── data/
│   ├── emails.csv           # Dados de treinamento
│   └── ANALISE_EMAIL_CLASSIFIER.md  # Documentação ML
├── tests/
│   ├── app/
│   │   ├── test_email_classifier.py
│   │   ├── test_schemas.py
│   │   └── test_views.py
│   └── test_main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── README.md
```

## 🔧 Comandos Make

```bash
# Executar aplicação
make runapp

# Executar aplicação em modo desenvolvimento
make runapp-dev

# Executar testes
make test

# Linting e formatação
make lint
```

## 📊 Modelo de Machine Learning

### Algoritmo
- **Multinomial Naive Bayes**: Classificador probabilístico
- **CountVectorizer**: Extração de características de texto
- **Pipeline**: Combinação de pré-processamento e classificação

### Características
- Processamento de texto automático
- Remoção de stopwords
- Normalização de texto
- Classificação binária (spam/ham)

### Performance
- Treinamento rápido
- Predições em tempo real
- Alta precisão em datasets balanceados

## 🔒 Segurança

- Validação de entrada com Pydantic
- Tratamento de erros robusto
- Logs de aplicação
- CORS configurado

## 🐛 Troubleshooting

### Problemas Comuns

1. **Modelo não encontrado**
   - Execute o treinamento primeiro
   - Verifique se o arquivo `model.pkl` existe

2. **Erro de dependências**
   - Atualize o pip: `pip install --upgrade pip`
   - Reinstale as dependências: `pip install -r requirements.txt`

3. **Porta em uso**
   - Mude a porta no docker-compose.yml
   - Ou use: `uvicorn app.main:app --port 8001`

## 📚 Documentação

- **Análise ML**: `data/ANALISE_EMAIL_CLASSIFIER.md`
- **API Docs**: `http://localhost:8000/docs`
- **Changelog**: `CHANGELOG.md`

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 📞 Suporte

Para suporte, abra uma issue no repositório ou entre em contato com a equipe de desenvolvimento.
