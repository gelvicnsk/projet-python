# prediction par regression logique
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Creer des données fictives
# x : caracteristiques (heures d'étude, heures de sommeil)

x = np.array([[5, 7], [8, 5], [6, 8], [9, 4], [10, 6], [4, 9], [3, 7], [7, 5], [2, 10], [1, 6]])

# y : labels (1 = reussite, 0 = échec)

y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# 2. diviser les données en ensemble d'entrainement et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# 3. Initialisation et entrainer le modèle
model = LogisticRegression()
model.fit(x_train, y_train)


# 4. Faire des predictions
y_pred = model.predict(x_test)

# 5. Evaluer le modèle 
accuracy= accuracy_score(y_test, y_pred)
print(f"Precision du modèle : {accuracy * 100: .2f}%")

# 6. Prédire pour les nouvelles données
# Par exemple, prédire pour un étudiant avec 8 heures déetude et 7 heures de sommeil
new_data = np.array([[8, 7]])
prediction = model.predict(new_data)
print(f"Prédiction pour les nouvelles données {new_data}: {prediction[0]}")