import pickle, gzip, numpy as np
import matplotlib.pyplot as plt

# Import des fonctions covVoisins etc.
import bayesien_naif as pps
import covvoisins as cvv
import MultinomialBN as mbn

# Load the dataset
f = gzip.open('./mnist.pkl.gz', 'rb')
u = pickle._Unpickler(f)
u.encoding = 'latin1'
p = u.load()
train_set, valid_set, test_set = p

# Bayesien

print()
print("# Naïf continu (exercice 3) :")
print()

# Récupération des images et des labels
images      = train_set[0]
images_test = test_set[0]

labels      = train_set[1]
labels_test = test_set[1]

class_con = pps.bn(10, images, labels)

print("L'erreur est de "
  + str(class_con.loss_rate(images_test, labels_test)) + "%"
  + " sur les " + " données de test")

## Naïf discret

print()
print("# Naïf discret (exercice 3) :")
print()

ims_binarize_train  = np.array(pps.binarize(images))
ims_binarize_test   = np.array(pps.binarize(images_test))

class_dis = pps.bn(10, images, labels)

print("L'erreur est de "
  + str(class_dis.loss_rate(ims_binarize_test, labels_test)) + "%"
  + " sur les données de test")

## Multinomial Bayesian Naive avec constante de lissage

print()
print("# Mutinomial Bayésien Naïf avec lissage : ici alpha = 0.1")
print()

clf = mbn.MultinomialBN(images, labels, images_test, labels_test, alpha = 0.00001)

print("L'erreur est de "
  + str(clf.error_rate()) + "%"
  + " sur les données de test")

## Multinomial Bayesian Naive avec constante de lissage (avec sklearn)
print()
print("# Mutinomial Bayésien Naïf avec lissage : ici alpha = 0.001 (sklearn)")
print()

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha = 0.001)
clf.fit(ims_binarize_train, labels)

def error_rate(images_test, labels_test):
    res = 0
    for i in range(len(images_test)):
        if (clf.predict(images_test[i:i+1]) != np.array([labels_test[i]])):
            res += 1
    return res/len(images_test) * 100

print("L'erreur est de "
  + str(error_rate(ims_binarize_test, labels_test)) + "%"
  + " sur les " + " données de test")

## Avec covariances

print()
print("# Covariances voisins (exercice 1) :")
print()

for i in range(class_dis.nm_classes):
  # Récupération des sous-ensembles d'images et moyennes par classe
  subtrain = images[labels == i]

  # Calcul des covariances des voisins et ajout aux covariances du modèle
  class_dis.variances[i] =  cvv.covVoisins(
                              subtrain,
                              class_dis.means[i], class_dis.variances[i]
                              )

  # Calcul des espérances avec les voisins et ajout aux moyennes
  class_dis.means[i] = cvv.meanVoisins(subtrain)

# Ajout des produits avec les voisins et ajout au jeu de données
cv_test = np.array([cvv.obsVoisins(i) for i in ims_binarize_test])

print("L'erreur est de "
  + str(class_dis.loss_rate(cv_test, labels_test)) + "%"
  + " sur les données de test")
