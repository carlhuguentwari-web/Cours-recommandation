import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cours = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'titre': [
        'Introduction à Python',
        'Machine Learning avec Python',
        'Analyse de données',
        'Deep Learning avancé',
        'Python pour le Web'
    ],
    'description': [
        'Apprenez les bases de Python',
        'Apprenez les algorithmes de ML avec Python',
        'Manipulez des données avec pandas et numpy',
        'Réseaux de neurones et CNN avec TensorFlow',
        'Utilisez Flask et Django pour créer des sites web'
    ]
})
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cours['description'])
similarité = cosine_similarity(tfidf_matrix)

def recommander(id_cours, n=3):
    index = cours[cours['id'] == id_cours].index[0]
    scores = list(enumerate(similarité[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommandations = [cours.iloc[i[0]] for i in scores[1:n+1]]
    return recommandations
st.set_page_config(page_title="Recommandation de cours", layout="centered")
st.markdown("<h1 style='color:#2c3e50;'>Recommandation de cours e-learning</h1>", unsafe_allow_html=True)
st.write("Sélectionnez un cours pour découvrir des suggestions similaires basées sur son contenu.")

choix = st.sidebar.selectbox("Choisissez un cours", cours['titre'])
id_choisi = cours[cours['titre'] == choix]['id'].values[0]
description_choisie = cours[cours['id'] == id_choisi]['description'].values[0]

st.markdown("<h3 style='color:#34495e;'>Cours sélectionné</h3>", unsafe_allow_html=True)
st.markdown(f"<strong>Titre:</strong> {choix}", unsafe_allow_html=True)
st.markdown(f"<strong>Description:</strong> {description_choisie}", unsafe_allow_html=True)


st.markdown("<h3 style='color:#34495e;'>Cours recommandés</h3>", unsafe_allow_html=True)
for cours_reco in recommander(id_choisi):
    st.markdown(f"<strong>Titre:</strong> {cours_reco['titre']}", unsafe_allow_html=True)
    st.markdown(f"<strong>Description:</strong> {cours_reco['description']}", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)


