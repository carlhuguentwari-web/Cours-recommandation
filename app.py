
import pandas as pd
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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cours['description'])
similarité = cosine_similarity(tfidf_matrix)

def recommander(id_cours, n=3):
    index = cours[cours['id'] == id_cours].index[0]
    scores = list(enumerate(similarité[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommandations = [cours.iloc[i[0]]['titre'] for i in scores[1:n+1]]
    return recommandations

import streamlit as st 
st.title(" Recommandation de cours e-learning")

choix = st.selectbox("Choisis un cours", cours['titre'])
id_choisi = cours[cours['titre'] == choix]['id'].values[0]

st.subheader("Cours recommandés:")
for titre in recommander(id_choisi):
    st.write(f" {titre}")
