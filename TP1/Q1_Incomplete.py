""""
Squelette de la question 1 du TP1 du cours LOG6308 donné par le Professeur Desmarais.
Écrit par Jean-Charles Layoun le 21 janvier 2023.
"""

import numpy as np
import pandas as pd 


# --------------------------------- Préparation des Données ---------------------------------
## Chargement des votes
votes = pd.read_csv('Data/votes.csv')

## Conversion du Pandas Datafram en Matrice Utilisateur Item
MUI = votes.pivot(index="user.id", columns="item.id", values="rating")
MUI.head()

## Convertir le DF à une matrice numpy
MUI_numpy      = MUI.to_numpy()
MUI_numpy_flat = MUI_numpy.reshape(-1)

## Création des indices pour les valeurs différentes de np.nan
indices    = np.arange(0, MUI_numpy.shape[0]*MUI_numpy.shape[1])
indices_na = indices[~np.isnan(MUI_numpy_flat)]

## Split Train Test des indices
nbre_replis = 5
np.random.shuffle(indices_na)
print(indices_na.shape)
idx_split = np.split(indices_na, nbre_replis)

## Je construis ma liste d'indice train et test
#  Pour faire une cross validation à 5 replis il suffit de remplacer 0 par i 
#   et itérer de 0 à 4
idx_train = np.delete(idx_split, 0, axis=0).flatten()
idx_test  = idx_split[0]

## J'enlève les valeurs de test de la matrice d'entrainement, et vice versa
MUI_numpy_flat_train = MUI_numpy_flat.copy()
MUI_numpy_flat_test  = MUI_numpy_flat.copy()
MUI_numpy_flat_train[idx_test] = np.nan
MUI_numpy_flat_test[idx_train] = np.nan
#  Je redonne la structure de matrice aux ensembles de test et d'entrainement
MUI_numpy_train = MUI_numpy_flat_train.reshape(MUI_numpy.shape)
MUI_numpy_test  = MUI_numpy_flat_test.reshape(MUI_numpy.shape)

# On s'assure d'avoir les bonnes dimensions
print(MUI_numpy_train.shape)
print(MUI_numpy_train[~np.isnan(MUI_numpy_train)].shape)
print(MUI_numpy_test[~np.isnan(MUI_numpy_test)].shape)


# --------------------------------- Définition des Métriques ---------------------------------
def RMSE_mat(y_pred, y_true):
    return np.sqrt(np.nanmean((y_pred - y_true)**2))

def MAE_mat(y_pred, y_true):
    return np.nanmean(np.abs(y_pred - y_true))
# Documentation pour np.nanmean : https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html
    
# --------------------------------- Prédictions des Valeurs Test ---------------------------------
# Prédiction des votes aléatoires entre 1 et 5 avec une distribution uniforme d'entiers
votes_alea_pred      = np.random.randint(1,6, MUI_numpy_train.shape)

# Vote Moyen
# Prédiction des votes grâce à la moyenne des votes de la matrice d'entrainement
votes_moyenne_pred   = np.nanmean(MUI_numpy_train)

# Vote Moyen Utilisateur
# Prédiction des votes grâce à la moyenne des votes par utilisateur de la matrice d'entrainement
votes_moyenne_U_pred = np.expand_dims(np.nanmean(MUI_numpy_train, axis=1), axis=-1)

# Vote Moyen Utilisateur
# Prédiction des votes grâce à la moyenne des votes par item de la matrice d'entrainement
votes_moyenne_I_pred = np.expand_dims(np.nanmean(MUI_numpy_train, axis=0), axis=0)

# Vote Moyen Attendu (Moyenne du vote moyen utilisateur et item)
# Méthode avec "outer addition"
votes_moyenne_A_pred = np.add.outer(votes_moyenne_U_pred[:,0], votes_moyenne_I_pred[0,:])/2
# Méthodes utilisant le broadcasting Numpy
# https://numpy.org/doc/stable/user/basics.broadcasting.html
votes_moyenne_A_pred = (votes_moyenne_U_pred + votes_moyenne_I_pred)/2

print("Résultats pour le vote aléatoire sur l'ensemble de test :")
print("RMSE: ", RMSE_mat(votes_alea_pred, MUI_numpy_test))
print("MAE: ", MAE_mat(votes_alea_pred, MUI_numpy_test),'\n')

print("Résultats pour le vote moyen sur l'ensemble de test :")
print("RMSE: ", RMSE_mat(votes_moyenne_pred, MUI_numpy_test))
print("MAE: ", MAE_mat(votes_moyenne_pred, MUI_numpy_test),'\n')

print("Résultats pour le vote moyen utilisateur sur l'ensemble de test :")
print("RMSE: ", RMSE_mat(votes_moyenne_U_pred, MUI_numpy_test))
print("MAE: ", MAE_mat(votes_moyenne_U_pred, MUI_numpy_test),'\n')

print("Résultats pour le vote moyen item sur l'ensemble de test :")
print("RMSE: ", RMSE_mat(votes_moyenne_I_pred, MUI_numpy_test))
print("MAE: ", MAE_mat(votes_moyenne_I_pred, MUI_numpy_test),'\n')

print("Résultats pour le vote moyen attendu sur l'ensemble de test :")
print("RMSE: ", RMSE_mat(votes_moyenne_A_pred, MUI_numpy_test))
print("MAE: ", MAE_mat(votes_moyenne_A_pred, MUI_numpy_test),'\n')


## Vous remarquerez le message d'alerte "Mean of empty slice" dans 
#  les résultats d'execution.
#  C'est parce qu'il existe 141 items qui ont uniquement un vote.
#  Donc si ce vote n'est pas présent dans l'ensemble d'entrainement
#  nous obtenons une colonne qu'avec que des NaN, ce qui nous renvoie 
#  le message d'alerte "Mean of empty slice".
#  Observer l'example ci-dessous pour comprendre comment le cas extrême 
#  est géré. 

# La moyenne d'une liste de NaN est NaN, 
# et cela même suite à une addition d'un nombre différent de nan
print(np.nanmean([np.nan, np.nan]) + 1)
# Donc le code dans ce fichier ne prend pas en compte les votes des items
# à un seul vote pour la prédiction avec le vote moyen item.