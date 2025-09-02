#Importando as bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

#importando o dataset de filmes
filmes = pd.read_csv('movies_metadata.csv', low_memory=False)
print(filmes.head(3))
print(filmes.columns)

#Separando as colunas
filmes = filmes[['id','original_title','original_language','vote_count']]
print(filmes.head(3))

#Verificando valores nulos
print(filmes.isnull().sum())

#Como possuem poucos valores nulos, optei por dropar as linhas correspondentes
filmes.dropna(inplace=True)
print(filmes.isnull().sum())

#Importando o dataset de avaliações
avaliacao = pd.read_csv('ratings.csv')
print(avaliacao.head(3))
print(avaliacao.columns)

#Separando as colunas que serão utilizadas
avaliacao = avaliacao[['userId','movieId','rating']]
print(avaliacao.head(3))

#Verificando valores nulos
print(avaliacao.isnull().sum())

#Verificando a média de avaliações por usuário
print(avaliacao['userId'].value_counts().mean())

#Separando Usuários com pelo menos 100 avaliações
qt_avaliacoes = avaliacao['userId'].value_counts() > 99
y = qt_avaliacoes[qt_avaliacoes].index

#Visualizando a quantidade de indices
print(avaliacao.shape)
print(y.shape)

#Seperando os usuários que pertencem a y
avaliacao = avaliacao.loc[y]

#Verificando o tamanho do dataset
print(avaliacao.shape)

#Separando filmes com pelo menos 1000 avaliações
filmes = filmes[filmes['vote_count'] > 999]
print(filmes.head(3))
print(filmes.shape)

#Verificando a quantidade de filmes por linguagem
filmes_linguagem = filmes['original_language'].value_counts()
print(filmes_linguagem.head(20))

#Selecionando apenas filmes em inglês
filmes = filmes[filmes['original_language'] == 'en']
print(filmes.head(20))

#Subistituindo o nome da coluna id para possuir o mesmo nome que em avaliações
filmes = filmes.rename(columns={'id':'movieId'})

#Verificando informações acerca dos datasets
print(filmes.info())
print(avaliacao.info())

#Modificando o tipo do id para int
filmes['movieId'] = filmes['movieId'].astype(int)

#Fundindo os datasets
avaliacoes_e_filmes = avaliacao.merge(filmes, on='movieId')

#Verificando informações do novo dataframe
print(avaliacoes_e_filmes.head(20))
print(avaliacoes_e_filmes.shape)
print(avaliacoes_e_filmes.isnull().sum())
print(avaliacoes_e_filmes.columns)

#Retirando campos com duplicidade
avaliacoes_e_filmes.drop_duplicates(['userId', 'original_title'], inplace=True)
avaliacoes_e_filmes.drop_duplicates(['userId', 'movieId'],inplace=True)
print(avaliacoes_e_filmes.shape)

#Excluindo a coluna movieId
del avaliacoes_e_filmes['movieId']

#Criando um PIVOT para transformar usuarios em coluna
filmes_pivot = avaliacoes_e_filmes.pivot(index='original_title', columns='userId', values='rating')
print(filmes_pivot.head(20))

#Preenchendo os valores nulos com zero
filmes_pivot.fillna(0, inplace=True)
print(filmes_pivot.head(20))

def nome_filmes():
    return avaliacoes_e_filmes['original_title']

#Importando csr_matrix do SciPy
from scipy.sparse import csr_matrix

#Transformando o dataset em uma matriz sparsa
filmes_sparse = csr_matrix(filmes_pivot)

#Importando KNN do SciKit
from sklearn.neighbors import NearestNeighbors

#Treinando o Modelo
model = NearestNeighbors(algorithm = 'brute')
model.fit(filmes_sparse)

#Testando o modelo utilizando um filme
distances, sugestions = model.kneighbors(filmes_pivot.filter(items = ['300'], axis = 0).values.reshape(1, -1))
for i in range(len(sugestions)):
    print(filmes_pivot.index[sugestions[i]],'\n')

#Antz
distances, sugestions = model.kneighbors(filmes_pivot.filter(items = ['Antz'], axis = 0).values.reshape(1, -1))
for i in range(len(sugestions)):
    print(filmes_pivot.index[sugestions[i]],'\n')

#Alien
distances, sugestions = model.kneighbors(filmes_pivot.filter(items = ['Alien'], axis = 0).values.reshape(1, -1))
for i in range(len(sugestions)):
    print(filmes_pivot.index[sugestions[i]],'\n')
indices_recomendados = sugestions[0][1:]
distancias_recomendadas = distances[0][1:]
filmes_recomendados = filmes_pivot.index[indices_recomendados]
epsilon = 0.05
dist_norm = (distancias_recomendadas - distancias_recomendadas.min()) / (distancias_recomendadas.max() - distancias_recomendadas.min())
dist_norm = dist_norm * (1 - 2*epsilon) + epsilon
similaridades = 1 - dist_norm

#Cores
cores = sns.color_palette("OrRd", len(filmes_recomendados))

# Plot
plt.figure(figsize=(8, 5))
plt.barh(filmes_recomendados[::-1], similaridades[::-1], color=cores[::-1])
plt.xlabel("Similaridade")
plt.title("Top Recomendações para 'Alien'")
plt.tight_layout()
plt.savefig("top_recomendacoes_alien.png", dpi=300)
plt.show()

#Casper
distances, sugestions = model.kneighbors(filmes_pivot.filter(items = ['Casper'], axis = 0).values.reshape(1, -1))
for i in range(len(sugestions)):
    print(filmes_pivot.index[sugestions[i]],'\n')

def escolha_filme(filme):
    distances, sugestions = model.kneighbors(filmes_pivot.filter(items=[filme], axis=0).values.reshape(1, -1))
    recomendation = []
    for i in range(len(sugestions)):
        for idx in sugestions[i]:
            filmes = filmes_pivot.index[idx]
            if filmes not in filme:
                recomendation.append(filmes_pivot.index[idx])

    return list(recomendation)
