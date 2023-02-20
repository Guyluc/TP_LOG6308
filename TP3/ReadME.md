# LOG6308 TP3 -- Approches contenu et techniques de graphes

![Python](https://img.shields.io/badge/Python-blue)![Jup](https://img.shields.io/badge/JupyterNotebook-orange)

## Consignes Générales du TP

Le travail doit être fait en **équipe de deux**. Vous trouverez l'énoncé du TP3 [ici](https://cours.polymtl.ca/MDesmarais/log6308/Tp/20231/tp3.html).

Il vous ait demander d'inclure dans votre soumission un fichier `Requirements.txt` qui contient toutes les dépendances des librairies Python utilisées pour résoudre le TP. 

Se référer au [README.MD](https://github.com/Learningchipmunk/TP_LOG6308_H23/blob/main/ReadME.md) pour savoir comment gérer son environnement Python. Rappel de la commande :

```bash
python -m venv envtp1
```

## Données à Télécharger

|       Fichier       |  Taille |                                                                                                                    Description                                                                                                                   |
|:-------------------:|:-------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [tp-matrix.dgt](https://cours.polymtl.ca/MDesmarais/log6308/Tp/20221/Tp2/tp-matrix.dgt)       | < 116Mo | Sous-ensemble de citations entre 50 497 articles extraits de la BD. Le format .mtx (Matrix Market Exchange) est utilisé pour les matrices creuses. En Python, on peut l'importer avec scipy.io import scipy.io M = scipy.io.mmread('matrix.dgt') |
| [12-articles.csv](https://cours.polymtl.ca/MDesmarais/log6308/Tp/20221/Tp2/12-articles.csv)     | < 1Ko   | Les 12 articles pour lesquels des recommandations doivent être fournies.                                                                                                                                                                         |
| [tp-matrix-names.csv](https://cours.polymtl.ca/MDesmarais/log6308/Tp/20221/Tp2/tp-matrix-names.csv) | 2Mo     | Liste identifiants du sous-ensemble de 50 498 articles.                                                                                                                                                                                          |
| [tp-abstracts.csv](https://cours.polymtl.ca/MDesmarais/log6308/Tp/20221/Tp2/tp-abstracts.csv)    | 55Mo    | Liste des résumés d'un sous-ensemble de 50 498 articles. Ces données ne sont pas exigées pour réaliser le TP3 mais fournies à toutes fins utiles.    

 ## License

Copyrights 2023 team LOG6308.

## Auteurs

- Professeur Michel Desmarais, [michel.desmarais@polypolymtl.ca](mailto:michel.desmarais@polymtl.ca).

- Jean-Charles LAYOUN, [jean-charles.layoun@polymtl.ca](mailto:jean-charles.layoun@polymtl.ca).
