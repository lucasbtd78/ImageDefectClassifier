# ImageDefectClassifier – Classification d’images de défauts

![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Python](https://img.shields.io/badge/Python-3.14-blue)

Ce projet contient un **CNN** construit avec **PyTorch** pour classifier des images de défauts en 2 classes : `class_A` et `class_B`.  
Il inclut le **dataloader**, le **modèle**, l’**entraînement** et la **prédiction sur un dossier d’images**.

---

## Structure du projet

```
ImageDefectClassifier/
│
├── data/ 
│ ├── class_A/
│ └── class_B/
├── predict_images/ 
├── dataset.py
├── CNN.py 
├── predict.py 
└── defect_cnn.pth
```
##  Dataset
Les images doivent être organisées dans ```data/``` par classe :
```
data/
├── class_A/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── class_B/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```
Le dataset est automatiquement split en train (80%) et validation (20%) dans ```dataset.py```.
- Class_A = label_0
- Class_B = label_1
- **Source des données :** https://www.kaggle.com/datasets/satishpaladi11/mechanic-component-images-normal-defected
## Entraînement
Pour entraîner le modèle, lancer :
``` python CNN.py ```
Le script :
- Charge le dataset (dataset.py)
- Entraîne le CNN 2 blocs avec BatchNorm, Dropout et AdaptivePooling
- Affiche la loss toutes les 10 itérations
- Évalue le modèle sur le jeu de validation à chaque epoch
- Sauvegarde le modèle entraîné dans defect_cnn.pth

## Prédiction
Pour prédire une image ou un dossier d’images :
``` python predict.py ```
- Place tes images à prédire dans ```predict_images/```
- Le script prédit chaque image et affiche :
```
img1.jpg → class_A
img2.jpg → class_B
```
## Modèle 
- **Architecture** : 2 blocs CNN
  - Conv2d → BatchNorm → ReLU → MaxPool
  - Conv2d → BatchNorm → ReLU → MaxPool
- **AdaptiveAvgPool2d** pour gérer la taille des features
- **Dropout 0.5** pour éviter l’overfitting
- **Fully Connected** : 32 → 2 (classes)

## Améliorations possibles
- Ajouter plus de blocs CNN pour un dataset plus large
- Data augmentation (rotation, flip, color jitter)
- Sauvegarde et chargement automatique des meilleurs modèles (```torch.save``` et ```torch.load```)
- Conversion du script predict.py en outil CLI avec arguments
