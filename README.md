# 🎥 Recomendação de Filmes com KNN

Este projeto implementa um sistema de recomendação de filmes utilizando o algoritmo K-Nearest Neighbors (KNN), treinado a partir de dados públicos do MovieLens e do dataset movies_metadata do TMDB.

O objetivo é sugerir filmes semelhantes a um título escolhido pelo usuário, com base nas avaliações de outros usuários.

📌 Funcionalidades

    Filtragem de filmes com base no número mínimo de votos.

    Filtragem de usuários com base no número mínimo de avaliações.

    Foco em filmes no idioma inglês.

    Criação de matriz esparsa para otimização de memória.

    Recomendações de filmes semelhantes utilizando KNN.

📂 Estrutura do Projeto

    movies_metadata.csv      # Dataset de metadados de filmes
    ratings.csv              # Dataset de avaliações de usuários
    Sistema_Recomendacao.py  # Código principal
    README.md                # Documentação do projeto

📊 Conjunto de Dados

    movies_metadata.csv (TMDB)
    Contém informações como id, original_title, original_language, vote_count.

    ratings.csv (MovieLens)
    Contém avaliações de usuários (userId, movieId, rating).

⚙️ Etapas do Projeto

    Importação e limpeza dos dados:

        Seleção apenas das colunas relevantes.

        Remoção de valores nulos.

        Filtro de usuários com pelo menos 100 avaliações.

        Filtro de filmes com pelo menos 1000 votos.

        Restrição a filmes no idioma inglês.


    Pré-processamento:

        Junção dos datasets de filmes e avaliações.

        Criação de uma matriz pivot com filmes como linhas e usuários como colunas.

        Substituição de valores nulos por zero.

        Conversão da matriz para formato esparso com csr_matrix.


    Treinamento do Modelo:

        Uso de NearestNeighbors do scikit-learn com algoritmo brute.

        Treinamento do modelo na matriz esparsa de filmes.


    Recomendações:

        Entrada de um título de filme.

        Retorno de uma lista de filmes similares com base nos vizinhos mais próximos.


🖥️ Exemplo de Uso

    Buscar filmes semelhantes a "300"
    distances, suggestions = model.kneighbors(
        filmes_pivot.filter(items=['300'], axis=0).values.reshape(1, -1)
    )
    for i in range(len(suggestions)):
        print(filmes_pivot.index[suggestions[i]])

    Saída esperada:

    Index(['300', '300: Rise of an Empire', 'Immortals', 'Clash of the Titans', 'Troy'], dtype='object', name='original_title')



📦 Bibliotecas Utilizadas

    Numpy

    Pandas

    Scikit-learn

    Scipy

