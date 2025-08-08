# 📧 Análise de Dados e Machine Learning: Classificação de Emails

## 🎯 Objetivo

Este documento demonstra o processo completo de análise de dados e construção de um modelo de machine learning para classificar emails como spam ou ham (legítimo). Vamos aprender os conceitos fundamentais de processamento de texto, extração de características e classificação usando Naive Bayes.

## 📚 Conceitos que vamos aprender

- **Processamento de Texto**: Limpeza e preparação de dados textuais
- **Extração de Características**: Conversão de texto em números
- **Classificação**: Algoritmos para categorizar dados
- **Naive Bayes**: Modelo probabilístico para classificação
- **Validação de Modelo**: Verificação da qualidade das predições

---

## 🔧 Importação das Bibliotecas

Primeiro, vamos importar as bibliotecas necessárias. Cada uma tem um papel específico:

- **pandas**: Manipulação e análise de dados
- **numpy**: Computação numérica
- **sklearn**: Algoritmos de machine learning
- **matplotlib/seaborn**: Visualização de dados
- **pickle**: Serialização de modelos

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
```

---

## 📖 Carregamento e Exploração dos Dados

### O que é Processamento de Texto?

Processamento de texto é o conjunto de técnicas para transformar dados textuais em formato que algoritmos de machine learning possam entender. Inclui:

- **Limpeza de texto**: Remoção de caracteres especiais, normalização
- **Tokenização**: Divisão do texto em palavras
- **Remoção de stopwords**: Palavras muito comuns que não agregam valor
- **Lematização**: Redução de palavras à sua forma base

### Por que é importante?

Texto bruto = Modelos ruins! O processamento adequado melhora significativamente a performance dos modelos de classificação.

```python
# Carregando os dados do arquivo CSV
data = pd.read_csv('emails.csv', sep='|', header=0, names=['label', 'message'])

# Visualizando as primeiras linhas
print("Primeiras 5 linhas dos dados:")
print(data.head())

# Informações sobre o dataset
print(f"\nDimensões: {data.shape}")
print(f"Colunas: {list(data.columns)}")
print(f"\nDistribuição das classes:")
print(data['label'].value_counts())
print(f"\nTipos de dados:")
print(data.dtypes)
```

---

## 🔄 Pré-processamento de Dados

### O que é Extração de Características?

Extração de características é o processo de converter texto em representações numéricas que algoritmos de machine learning possam usar.

### Técnicas de Extração de Características:

#### 1. **CountVectorizer** (Bag of Words)
- **Conceito**: Conta a frequência de cada palavra no texto
- **Vantagem**: Simples e eficaz
- **Limitação**: Não considera ordem das palavras
- **Exemplo**: "Win free iPhone" → [1, 1, 1, 0, 0, ...]

#### 2. **TF-IDF** (Term Frequency-Inverse Document Frequency)
- **Conceito**: Considera a importância das palavras no corpus
- **Vantagem**: Reduz peso de palavras muito comuns
- **Limitação**: Mais complexo computacionalmente

### Por que fazer isso?
- **Algoritmos numéricos**: Machine learning trabalha com números
- **Padronização**: Textos de diferentes tamanhos ficam comparáveis
- **Redução de dimensionalidade**: Remove ruído e melhora performance

```python
# Preparando os dados
def prepare_data(data):
    """Prepara os dados para treinamento."""
    df = data.copy()
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    return df

# Aplicando preparação
df = prepare_data(data)

print("Dados preparados:")
print(f"Spam (1): {df['label'].value_counts()[1]} emails")
print(f"Ham (0): {df['label'].value_counts()[0]} emails")
print(f"\nExemplo de mensagem:")
print(df['message'].iloc[0])
```

---

## 🔍 Análise Exploratória dos Dados

### Distribuição das Classes

É importante verificar se os dados estão balanceados entre as classes:

```python
# Visualizando distribuição das classes
plt.figure(figsize=(8, 6))
df['label'].value_counts().plot(kind='bar')
plt.title('Distribuição de Classes')
plt.xlabel('Classe (0=Ham, 1=Spam)')
plt.ylabel('Quantidade')
plt.show()

print("Distribuição das classes:")
print(df['label'].value_counts())
print(f"\nProporção:")
print(df['label'].value_counts(normalize=True))
```

### Análise de Palavras Frequentes

Vamos identificar as palavras mais comuns em spam vs ham:

```python
from collections import Counter
import re

def get_word_frequencies(texts, label):
    """Extrai frequência de palavras para uma classe."""
    all_words = []
    for text in texts:
        # Limpeza básica
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)
    
    return Counter(all_words)

# Palavras mais frequentes em spam
spam_texts = df[df['label'] == 1]['message']
spam_words = get_word_frequencies(spam_texts, 'spam')

# Palavras mais frequentes em ham
ham_texts = df[df['label'] == 0]['message']
ham_words = get_word_frequencies(ham_texts, 'ham')

print("Top 10 palavras em SPAM:")
for word, count in spam_words.most_common(10):
    print(f"{word}: {count}")

print("\nTop 10 palavras em HAM:")
for word, count in ham_words.most_common(10):
    print(f"{word}: {count}")
```

---

## 🤖 Construção do Modelo de Machine Learning

### O que é Naive Bayes?

Naive Bayes é um algoritmo de classificação baseado no teorema de Bayes. É "naive" porque assume independência entre as características.

### Fórmula Matemática:
**P(Classe|Características) ∝ P(Características|Classe) × P(Classe)**

Onde:
- **P(Classe|Características)**: Probabilidade da classe dado as características
- **P(Características|Classe)**: Probabilidade das características dado a classe
- **P(Classe)**: Probabilidade a priori da classe

### Por que Naive Bayes?
- **Simples e rápido**: Computacionalmente eficiente
- **Bom baseline**: Funciona bem para classificação de texto
- **Poucos dados**: Funciona mesmo com datasets pequenos
- **Interpretável**: Fácil de entender as predições

```python
# Criando o pipeline de classificação
from sklearn.pipeline import Pipeline

# CountVectorizer para extração de características
vectorizer = CountVectorizer(
    max_features=1000,  # Limita número de características
    stop_words='english',  # Remove stopwords
    ngram_range=(1, 2)  # Considera palavras e bigramas
)

# MultinomialNB para classificação
classifier = MultinomialNB()

# Pipeline combinando vectorizer e classificador
pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', classifier)
])

print("Pipeline criado com sucesso!")
print("Etapas:")
print("1️⃣  CountVectorizer: Extração de características")
print("2️⃣  MultinomialNB: Classificação")
```

---

## 📊 Treinamento e Avaliação do Modelo

### Divisão dos Dados

É crucial dividir os dados em treino e teste para avaliar a performance real do modelo:

```python
# Dividindo os dados
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Divisão dos dados:")
print(f"Treino: {X_train.shape[0]} emails")
print(f"Teste: {X_test.shape[0]} emails")
print(f"\nDistribuição no treino:")
print(y_train.value_counts())
print(f"\nDistribuição no teste:")
print(y_test.value_counts())
```

### Treinamento do Modelo

```python
# Treinando o modelo
pipeline.fit(X_train, y_train)

print("✅ Modelo treinado com sucesso!")

# Fazendo predições no conjunto de teste
y_pred = pipeline.predict(X_test)

print(f"\nPredições realizadas: {len(y_pred)}")
print(f"Spam preditos: {sum(y_pred == 1)}")
print(f"Ham preditos: {sum(y_pred == 0)}")
```

### Avaliação do Modelo

```python
# Relatório de classificação
print("📊 Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()

print("\n📈 Métricas de Performance:")
print(f"Precisão: {cm[1,1] / (cm[1,1] + cm[0,1]):.3f}")
print(f"Recall: {cm[1,1] / (cm[1,1] + cm[1,0]):.3f}")
print(f"F1-Score: {2 * cm[1,1] / (2 * cm[1,1] + cm[0,1] + cm[1,0]):.3f}")
```

---

## 🔍 Análise de Erros

### O que são Falsos Positivos e Falsos Negativos?

- **Falso Positivo**: Email legítimo classificado como spam
- **Falso Negativo**: Spam classificado como legítimo

### Por que analisar erros?
- **Melhorar o modelo**: Identificar padrões problemáticos
- **Ajustar threshold**: Balancear precisão vs recall
- **Feature engineering**: Criar características mais discriminativas

```python
# Identificando erros de classificação
errors = X_test[y_test != y_pred]
error_labels = y_test[y_test != y_pred]
error_predictions = y_pred[y_test != y_pred]

print("🔍 Análise de Erros:")
print(f"Total de erros: {len(errors)}")

# Falsos positivos (ham classificado como spam)
fp_indices = (y_test == 0) & (y_pred == 1)
fp_count = sum(fp_indices)

# Falsos negativos (spam classificado como ham)
fn_indices = (y_test == 1) & (y_pred == 0)
fn_count = sum(fn_indices)

print(f"Falsos Positivos (Ham → Spam): {fp_count}")
print(f"Falsos Negativos (Spam → Ham): {fn_count}")

# Exemplos de erros
if len(errors) > 0:
    print(f"\n📝 Exemplos de erros:")
    for i, (text, true_label, pred_label) in enumerate(zip(errors, error_labels, error_predictions)):
        if i < 3:  # Mostra apenas os primeiros 3
            label_names = {0: 'Ham', 1: 'Spam'}
            print(f"Texto: {text[:100]}...")
            print(f"Real: {label_names[true_label]}, Predito: {label_names[pred_label]}")
            print("-" * 50)
```

---

## 🚀 Modelo Final e Predições

### Salvando o Modelo

```python
# Salvando o modelo treinado
with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("💾 Modelo salvo com sucesso!")
```

### Função de Predição

```python
def predict_email(message):
    """
    Prediz se um email é spam ou ham.
    
    Args:
        message (str): Texto do email
        
    Returns:
        str: 'spam' ou 'ham'
    """
    # Carregando o modelo
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Fazendo predição
    prediction = model.predict([message])[0]
    
    return 'spam' if prediction == 1 else 'ham'

# Testando o modelo
test_messages = [
    "Win a free iPhone now! Click here!",
    "Hello, how are you? Let's meet tomorrow.",
    "URGENT: Your account has been suspended!",
    "Please review the attached document.",
    "Free money! Make $1000/day from home!"
]

print("🧪 Testando o Modelo:")
for message in test_messages:
    prediction = predict_email(message)
    print(f"'{message[:50]}...' → {prediction}")
```

---

## 📊 Análise de Probabilidades

### Por que analisar probabilidades?

As probabilidades nos dão mais informação sobre a confiança do modelo:

```python
def predict_with_probability(message):
    """
    Prediz com probabilidade de confiança.
    """
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Probabilidades para cada classe
    probabilities = model.predict_proba([message])[0]
    
    prediction = 'spam' if probabilities[1] > probabilities[0] else 'ham'
    confidence = max(probabilities)
    
    return prediction, confidence

# Testando com probabilidades
print("📊 Predições com Probabilidade:")
for message in test_messages:
    pred, conf = predict_with_probability(message)
    print(f"'{message[:40]}...' → {pred} (confiança: {conf:.3f})")
```

---

## 🎓 Resumo do que Aprendemos

### 📊 **Processamento de Texto**
- **Importância**: Texto limpo = modelos melhores
- **Técnicas**: CountVectorizer, TF-IDF
- **Ferramentas**: NLTK, spaCy (para processamento avançado)

### 🤖 **Machine Learning**
- **Algoritmo**: Multinomial Naive Bayes
- **Objetivo**: Classificar emails como spam/ham
- **Aplicação**: Filtros de spam, moderação de conteúdo

### 🔍 **Validação de Modelo**
- **Métricas**: Precisão, Recall, F1-Score
- **Matriz de confusão**: Análise de erros
- **Probabilidades**: Confiança das predições

### 🚀 **Próximos Passos**
- Experimentar outros algoritmos (SVM, Random Forest)
- Feature engineering mais avançado (word embeddings)
- Validação cruzada
- Otimização de hiperparâmetros

### 💡 **Conceitos-Chave**
1. **Texto é diferente**: Precisa de processamento especial
2. **Extração de características é crucial**: Boas features = bom modelo
3. **Balanceamento importa**: Classes desbalanceadas afetam performance
4. **Interpretabilidade é útil**: Entender por que o modelo classifica

---

**🎯 Lembre-se**: Classificação de texto é uma área rica em possibilidades. Comece simples, valide sempre, melhore gradualmente!

---

## 📝 Código Completo

```python
# Importação das bibliotecas
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle

# Carregamento dos dados
data = pd.read_csv('emails.csv', sep='|', header=0, names=['label', 'message'])

# Preparação dos dados
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Divisão dos dados
X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criação do pipeline
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
classifier = MultinomialNB()
pipeline = Pipeline([('vectorizer', vectorizer), ('classifier', classifier)])

# Treinamento do modelo
pipeline.fit(X_train, y_train)

# Avaliação
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Salvando o modelo
with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Predição
def predict_email(message):
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    prediction = model.predict([message])[0]
    return 'spam' if prediction == 1 else 'ham'

# Teste
result = predict_email("Win a free iPhone now!")
print(f"Predição: {result}")
``` 