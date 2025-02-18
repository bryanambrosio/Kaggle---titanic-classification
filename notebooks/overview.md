# Visão Geral do Código

## Objetivo
Este notebook tem como objetivo construir um modelo de aprendizado de máquina para prever a sobrevivência de passageiros no Titanic com base em dados disponíveis.

## Etapas do Código

### 1. Importação de Bibliotecas
```python
import numpy as np  # Biblioteca para operações numéricas
import pandas as pd  # Biblioteca para manipulação de dados
import joblib  # Para salvar e carregar modelos treinados
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

### 2. Carregamento de Dados
```python
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
```

### 3. Pré-processamento dos Dados
- Remoção de colunas irrelevantes como `Ticket`, `Cabin`, `Name` e `PassengerId`.
- Tratamento de valores ausentes:
  ```python
  train_data.fillna(train_data.median(), inplace=True)
  test_data.fillna(test_data.median(), inplace=True)
  ```
- Codificação de variáveis categóricas usando `LabelEncoder`.

### 4. Divisão dos Dados
```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 5. Treinamento do Modelo
```python
clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
```

### 6. Avaliação do Modelo
```python
predictions = clf.predict(X_val)
accuracy = accuracy_score(y_val, predictions)
print(f"Acurácia do modelo: {accuracy:.4f}")
```

### 7. Salvamento do Modelo
```python
joblib.dump(clf, 'titanic_model.pkl')
```

## Resultados Esperados
O código gera um modelo que pode prever a sobrevivência de passageiros com base nas características fornecidas. A acurácia do modelo é calculada para avaliar seu desempenho.

---
Este notebook pode ser expandido com ajustes nos hiperparâmetros do modelo, engenharia de recursos e testes com diferentes algoritmos de aprendizado de máquina.

