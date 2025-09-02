from Sistema_Recomendacao import escolha_filme, nome_filmes
import streamlit as st
import pandas as pd

def verificar_poster(filme):
    if titulo.loc[titulo['Title'] == filme].iloc[0]['Poster_URL'] == '':
     return None
    else:
        img = titulo.loc[titulo['Title'] == filme].iloc[0]['Poster_URL']
        return img

filmes = nome_filmes()
filmes = pd.DataFrame({'Title': filmes})
posters = pd.read_csv('posters.csv', sep=';')
posters = posters[['Title','Poster_URL']]
titulo = filmes.merge(posters, on='Title', how='left')
titulo.fillna('', inplace=True)

x = st.selectbox('Selecione o filme', titulo)
if verificar_poster(x) is not None:
    st.image(verificar_poster(x))

st.write("Com base no filme selecionado, recomendo os seguintes filmes:")

recomendacao = escolha_filme(x)
for filme in recomendacao:
    if verificar_poster(filme) is not None:
        st.write(filme)
        st.image(verificar_poster(filme))
    else:
        st.write(filme)
        st.write("Poster Indisponível")