# %% Importation des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
df = pd.read_csv('C:/Users/gelvi/OneDrive - umontpellier.fr/Documents/DATA/Jeux de données/ozone.csv')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Aperçu des données
print("Aperçu des données :")
print(df.head())
print("\nRésumé statistique :")
print(df.describe())
print("\nInformations sur les colonnes :")
print(df.info())

# %% Exploration des données
# Vérification des valeurs manquantes
missing_values = df.isnull().sum()
print("\nValeurs manquantes par colonne :")
print(missing_values)

# Distribution des lectures d'ozone
plt.figure(figsize=(8, 5))
sns.histplot(df['ozone_reading'], kde=True, color='blue')
plt.title("Distribution des lectures d'ozone")
plt.xlabel("Lecture d'ozone")
plt.ylabel("Fréquence")
plt.show()

# %% Visualisation des tendances mensuelles
plt.figure(figsize=(10, 6))
sns.boxplot(x='Month', y='ozone_reading', data=df, palette='Set3')
plt.title("Distribution des lectures d'ozone par mois")
plt.xlabel("Mois")
plt.ylabel("Lecture d'ozone")
plt.show()

# %% Prétraitement des données
# Remplir les valeurs manquantes (exemple : remplir par la moyenne)
df['ozone_reading'] = df['ozone_reading'].fillna(df['ozone_reading'].mean())
df['pressure_height'] = df['pressure_height'].fillna(df['pressure_height'].mean())

# Encodage des jours de la semaine (si nécessaire)
df['Day_of_week'] = pd.Categorical(df['Day_of_week']).codes

# Ajout d'une colonne pour les saisons
def assign_season(month):
    if month in [12, 1, 2]:
        return "Hiver"
    elif month in [3, 4, 5]:
        return "Printemps"
    elif month in [6, 7, 8]:
        return "Été"
    else:
        return "Automne"

df['Season'] = df['Month'].apply(assign_season)

# %% Analyse de corrélation
corr_matrix = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Matrice de corrélation")
plt.show()

# %% Visualisation des lectures d'ozone en fonction de la hauteur de pression
plt.figure(figsize=(10, 6))
sns.scatterplot(x='pressure_height', y='ozone_reading', data=df, alpha=0.6)
plt.title("Lecture d'ozone vs Hauteur de pression")
plt.xlabel("Hauteur de pression")
plt.ylabel("Lecture d'ozone")
plt.show()

# %% Sauvegarde des données prétraitées
output_path = 'C:/Users/gelvi/OneDrive - umontpellier.fr/Documents/DATA/Jeux de données/ozone_cleaned.csv'
df.to_csv(output_path, index=False)
print(f"Données nettoyées sauvegardées dans {output_path}")




# Pediction par regression lineaire
# %% Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Charger les données nettoyées
df = pd.read_csv('C:/Users/gelvi/OneDrive - umontpellier.fr/Documents/DATA/Jeux de données/ozone_cleaned.csv')

# %% Sélection des variables (features) et de la cible (ozone_reading)
X = df[['Month', 'Day_of_week', 'pressure_height']]  # Variables explicatives
y = df['ozone_reading']  # Variable cible

# %% Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Normalisation des données (facultatif mais recommandé pour certains modèles)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% Entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# %% Prédictions sur l'ensemble de test
y_pred = model.predict(X_test_scaled)

# %% Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Erreur quadratique moyenne (MSE) : {mse:.2f}")
print(f"Racine de l'erreur quadratique moyenne (RMSE) : {rmse:.2f}")
print(f"Score R² : {r2:.2f}")

# %% Visualisation des prédictions
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title("Prédictions des lectures d'ozone vs Réelles")
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions")
plt.show()

# %% Sauvegarde du modèle
import joblib
joblib.dump(model, 'ozone_prediction_model.pkl')
print("Modèle de prédiction sauvegardé sous 'ozone_prediction_model.pkl'")
