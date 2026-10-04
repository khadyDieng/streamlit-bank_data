"""
Streamlit — Souscription à un dépôt à terme (TP2, Bank_Data)
Construite sur le modèle d'application fourni par le prof : les objets sauvegardés
dans le notebook sont rechargés, puis le meilleur des sept classifieurs fait la prédiction.

En local :  streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st

st.set_page_config(page_title="Dépôt à terme", page_icon="💳", layout="centered")


# ---------- Objets issus du notebook (chargés une seule fois) ----------
@st.cache_resource
def charger_objets():
    encoders = jb.load("encoders.joblib")   # 9 variables texte + la cible y
    uniques = jb.load("uniques.joblib")     # modalités de chaque variable texte
    scaler = jb.load("scaler.joblib")       # StandardScaler
    modele = jb.load("best_model.joblib")   # classifieur retenu
    return encoders, uniques, scaler, modele


encoders, uniques, scaler, modele = charger_objets()
reponses = uniques[-1]  # no / yes
TRADUCTION = {"no": "Le client ne souscrit pas", "yes": "Le client souscrit"}


# ---------- Prédiction pour un client ----------
def Pred_func(age, job, marital, education, housing, loan, contact, month,
              day_of_week, duration, campaign, pdays, previous, poutcome):
    textes = [job, marital, education, housing, loan, contact, month, day_of_week, poutcome]
    codes = [encoders[i].transform([valeur])[0] for i, valeur in enumerate(textes)]
    c_job, c_marital, c_education, c_housing, c_loan, c_contact, c_month, c_day, c_poutcome = codes
    # même ordre de colonnes que dans le notebook
    vecteur = np.array([age, c_job, c_marital, c_education, c_housing, c_loan, c_contact, c_month,
                        c_day, duration, campaign, pdays, previous, c_poutcome]).reshape(1, -1)
    vecteur_norm = scaler.transform(vecteur)
    classe = modele.predict(vecteur_norm)[0]
    return reponses[classe]


# ---------- Prédiction pour un fichier ----------
def Pred_func_csv(fichier):
    tableau = pd.read_csv(fichier)
    resultats = []
    for ligne in tableau.values:
        resultats.append(Pred_func(*ligne[:14]))
    tableau["y prédit"] = resultats
    return tableau


st.title("💳 Souscription à un dépôt à terme")
st.caption("Le client contacté va-t-il souscrire ? Prédiction à partir de son profil et de l'appel.")
onglet_un, onglet_csv = st.tabs(["Un client", "Fichier CSV"])

with onglet_un:
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Âge", 17, 100, 40)
        job = st.selectbox("Profession", list(uniques[0]))
        marital = st.selectbox("Situation familiale", list(uniques[1]))
        education = st.selectbox("Études", list(uniques[2]))
    with c2:
        housing = st.selectbox("Crédit immobilier", list(uniques[3]))
        loan = st.selectbox("Crédit personnel", list(uniques[4]))
        contact = st.selectbox("Canal de contact", list(uniques[5]))
        poutcome = st.selectbox("Issue de la campagne précédente", list(uniques[8]))
    with c3:
        month = st.selectbox("Mois de l'appel", list(uniques[6]))
        day_of_week = st.selectbox("Jour de l'appel", list(uniques[7]))
        duration = st.number_input("Durée de l'appel (secondes)", 0, 5000, 250)
        campaign = st.number_input("Appels pendant la campagne", 1, 60, 2)
    c4, c5 = st.columns(2)
    with c4:
        pdays = st.number_input("Jours depuis le dernier contact (999 = jamais contacté)", 0, 999, 999)
    with c5:
        previous = st.number_input("Contacts lors des campagnes précédentes", 0, 10, 0)

    if st.button("Prédire", type="primary", use_container_width=True):
        try:
            reponse = Pred_func(age, job, marital, education, housing, loan, contact, month,
                                day_of_week, duration, campaign, pdays, previous, poutcome)
            resultat = TRADUCTION.get(reponse, reponse)
            st.success(f"**Décision estimée :** {resultat}")
        except Exception as erreur:
            st.error(f"Prédiction impossible : {erreur}")
with onglet_csv:
    st.info("Colonnes attendues, dans cet ordre : age, job, marital, education, housing, loan, contact, "
            "month, day_of_week, duration, campaign, pdays, previous, poutcome.")
    fichier = st.file_uploader("Choisir un fichier CSV", type="csv")
    if fichier is not None:
        try:
            tableau = Pred_func_csv(fichier)
            st.dataframe(tableau, use_container_width=True)
            st.download_button("Télécharger les résultats", tableau.to_csv(index=False).encode("utf-8"),
                               "resultats_clients.csv", "text/csv")
        except Exception as erreur:
            st.error(f"Fichier non traité : {erreur}")
