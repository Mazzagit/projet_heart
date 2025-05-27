import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Charger les données
df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Sauvegarde du modèle
joblib.dump(model, "model_heart.joblib")
print("Modèle sauvegardé.")
