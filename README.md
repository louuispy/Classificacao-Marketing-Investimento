# Classificacao-Marketing-Investimento

Faaaaaaaaaaaaala galera, tudo bem com vocês? Espero que sim!  
  
Venho aqui compartilhar com vocês um projeto que eu fiz recentemente, que consiste basicamente em uma análise exploratória de um dataset com dados sobre clientes de uma empresa e a criação de um algoritmo de Machine Learning para prever se o cliente irá aderir ao investimento proposto por uma empresa. É um projeto voltado para Machine Learning e Ciência de Dados, principalmente no que se refere à criação de algoritmos de classificação. Espero que gostem! 😊

---

## Construção do projeto

### 1. Importando as bibliotecas e lendo a base de dados

De início, nós realizamos a importação das bibliotecas e fazemos a leitura da base de dados. As bibliotecas usadas de início foram:
- `pandas`
- `numpy`
- `plotly`
  
Feito isso, fazemos uma leitura da base de dados, e logo em seguida visualizamos usando o comando:

````python
dados = pd.read_csv('datasets/marketing_investimento.csv')
dados.head()

````

Após isso, fazemos uma análise de dados inicial acerca do dataset, e logo em seguida, fazemos uma análise mais aprofundada, visualizando os dados por meio de gráficos, utilizando a biblioteca Plotly. 

![image](https://github.com/user-attachments/assets/0c0f8b85-d68d-4dce-8b7c-9f56fc7d8f8d)

![image](https://github.com/user-attachments/assets/0ae0f2f2-e379-46f3-81d0-766996668f72)

![image](https://github.com/user-attachments/assets/9b341c4e-79d4-46fb-9270-dc7845a9fe90)

![image](https://github.com/user-attachments/assets/7e5ebb8d-7451-49d2-a12c-d2cf65e28ef7)

![image](https://github.com/user-attachments/assets/2ee5c28e-15e9-4909-8f87-be1c8920b831)


Feito isso, descobrimos que não haviam outliers nos dados. Com isso, podemos prosseguir para a próxima etapa.

----

### 2. OneHotEncoding dos dados explicativos e LabelEncoding da variável target

Após essa etapa de análise exploratória, fizemos um OneHotEncoding dos dados explicativos (X), e o LabelEncoding dos dados da variável target (y), para assim podermos usar esses dados no treinamento dos algoritmos de Machine Learning.
Para isso usamos o seguinte codigo:

````python
from sklearn.compose import make_column_transformer # Faz transformações na base de dados
from sklearn.preprocessing import OneHotEncoder

````

````python
one_hot = make_column_transformer((
    OneHotEncoder(drop='if_binary'), # Remove uma coluna se ela for binária, ou seja, se possuir apenas 2 categorias
    ['estado_civil', 'escolaridade', 'inadimplencia', 'fez_emprestimo']),
    remainder='passthrough',      # Impede que as colunas numéricas sejam removidas (por padrão)
    sparse_threshold=0)           # Permite que as colunas mantenham todos os valores

X = one_hot.fit_transform(X)
````

````python
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
````

Agora, podemos criar os algoritmos de Machine Learning.

----

### 3. Criando os algoritmos de Machine Learning

Criamos no total 3 algoritmos de Machine Learning, e ao final, salvamos com o Pickle e testamos com dados novos.
Os algoritmos criados foram:
- `DummyClassifier`
- `DecisionTreeClassifier`
- `KNN`

### DummyClassifier

````python
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier() 
dummy.fit(X_treino, y_treino)

dummy.score(X_teste, y_teste)
````
Acurácia: 60%

### DecisionTreeClassifier (com `max_depth`)

````python
from sklearn.tree import DecisionTreeClassifier
arvore = DecisionTreeClassifier(max_depth=3, random_state=5)
arvore.fit(X_treino, y_treino)

arvore.score(X_teste, y_teste)
````

Acurácia: 71%


****Visualização da árvore de decisões****

![image](https://github.com/user-attachments/assets/b4698647-3199-4adb-8908-c4940fc89a5c)

### KNN

Para o KNN, primeiro fizemos a normalização dos dados.

````python
from sklearn.preprocessing import MinMaxScaler

normalizacao = MinMaxScaler()

X_treino_normalizado = normalizacao.fit_transform(X_treino)
X_teste_normalizado = normalizacao.transform(X_teste)

````

Após esse processo, criamos o algoritmo KNN.

````python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_treino_normalizado, y_treino)
knn.score(X_teste_normalizado, y_teste)
````

Acurácia: 68%

----

### Comparando os modelos e salvando o melhor modelo

Após realizar todo esse processo, descobrimos que o melhor modelo é o DecisionTreeClassifier, uma vez que apresentou melhor acurácia. Vamos salvar tanto este modelo, quanto o OneHotEncoder, para que o modelo funcione corretamente.

````python
import pickle

with open('modelo_onehotenc.pkl', 'wb') as arquivo: 
    pickle.dump(one_hot, arquivo)

with open('modelo_arvore.pkl', 'wb') as arquivo: 
    pickle.dump(arvore, arquivo) 
````

----

### Testando o modelo com dados novos

Após todo esse processo, testamos os seguintes dados com o modelo:

````python
novo_dado = {
    'idade': [45], 
    'estado_civil': ['solteiro (a)'], 
    'escolaridade': ['superior'], 
    'inadimplencia': ['nao'], 
    'saldo': [2300], 
    'fez_emprestimo': ['nao'], 
    'tempo_ult_contato': [800], 
    'numero_contatos': [4]
}
````
E logo em seguida, convertemos esse dado para um DataFrame.

````python
novo_dado = pd.DataFrame(novo_dado) 
````

E por fim, fizemos a leitura dos modelos que salvamos com o pickle, e fizemos os testes.

````python
modelo_one_hot = pd.read_pickle('modelo_onehotenc.pkl')
modelo_arvore = pd.read_pickle('modelo_arvore.pkl')
````
````python
novo_dado = modelo_one_hot.transform(novo_dado)
modelo_arvore.predict(novo_dado) 
````
Ao final, obtivemos uma classificação de [1], indicando que o cliente irá aderir ao investimento.

----

E esse foi o projeto gente, espero que tenham gostado! ^^
