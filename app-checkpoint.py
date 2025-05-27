import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Configuration
st.set_page_config(page_title="💖 Analyse Cardiaque", layout="centered")

# CSS pour une touche féminine
st.markdown("""
    <style>
   body {
        background-color: #8b4513;  
    }
    .main {
        background-color: #8b4513;
    }
    h1, h2 {
        color: #c71585;
    }
    </style>
""", unsafe_allow_html=True)

# Chargement des données
df = pd.read_csv("heart.csv")
model = joblib.load("model_heart.joblib")

# Sidebar
st.sidebar.title("🔍 Navigation")
page = st.sidebar.selectbox("Choisir une section", ["Aperçu des données", "Analyse exploratoire", "Prédiction"])

# Page : Aperçu des données
if page == "Aperçu des données":
    st.title("📊 Aperçu des données")
    st.write(df.head())
    st.write("Dimensions :", df.shape)
    st.write("Colonnes :", df.columns.tolist())
    st.dataframe(df.head())

# Page : Analyse exploratoire
elif page == "Analyse exploratoire":
    st.title("🔬 Analyse Exploratoire")
    st.markdown("Voici une matrice de corrélation pour comprendre les liens entre les variables.")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Page : Prédiction
elif page == "Prédiction":
    st.title("❤️ Prédiction de Maladie Cardiaque")
    st.markdown("Entrez les données du patient :")
    
    # Saisie utilisateur
    age = st.slider("Âge", 20, 100, 50)
    sex = st.selectbox("Sexe", [0, 1])
    cp = st.selectbox("Douleur Poitrine (cp)", [0, 1, 2, 3])
    trestbps = st.slider("Pression artérielle", 80, 200, 120)
    chol = st.slider("Cholestérol", 100, 600, 200)
    fbs = st.selectbox("FBS > 120 mg/dl", [0, 1])
    restecg = st.selectbox("ECG", [0, 1, 2])
    thalach = st.slider("Fréquence cardiaque max", 60, 220, 150)
    exang = st.selectbox("Angine d'effort", [0, 1])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("Pente ST", [0, 1, 2])
    ca = st.selectbox("Nombre de vaisseaux colorés", [0, 1, 2, 3])
    thal = st.selectbox("Thal", [0, 1, 2, 3])
    
    features = [[age, sex, cp, trestbps, chol, fbs, restecg,
                 thalach, exang, oldpeak, slope, ca, thal]]
    
    # Bouton prédire
    if st.button("Prédire"):
        prediction = model.predict(features)[0]

        if prediction == 1:
            st.success("✅ Risque détecté : la personne est susceptible d’avoir une maladie cardiaque.")
        else:
            st.info("💖 Pas de risque détecté : la personne est en bonne santé cardiaque.")
        
        st.markdown("---")
