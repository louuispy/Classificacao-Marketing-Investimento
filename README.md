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

![image](https://github.com/user-attachments/assets/9b341c4e-79d4-46fb-9270-dc7845a9fe90)

![image](https://github.com/user-attachments/assets/7e5ebb8d-7451-49d2-a12c-d2cf65e28ef7)

![image](https://github.com/user-attachments/assets/2ee5c28e-15e9-4909-8f87-be1c8920b831)


Feito isso, descobrimos que não haviam outliers nos dados. Com isso, podemos prosseguir para a próxima etapa.

### 2. OnHotEncoding dos dados explicativos e LabelEncoding da variável target

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

