import pickle, gzip, numpy as np

# Import des fonctions covVoisins etc.
import bayesien_naif_discret as pps
import covvoisins as cvv

# Load the dataset
f = gzip.open('./mnist.pkl.gz', 'rb')
u = pickle._Unpickler(f)
u.encoding = 'latin1'
p = u.load()
train_set, valid_set, test_set = p

## Bayesien

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
  + " sur les " + str(len(images)) + " données de test")

## Naïf discret

print()
print("# Naïf discret (exercice 3) :")
print()

ims_binarize_train  = np.array(pps.binarize(images))
ims_binarize_test   = np.array(pps.binarize(images_test))

class_dis = pps.bn(10, images, labels)

alpha = 0.00001

print("L'erreur est de "
  + str(class_dis.loss_rate(ims_binarize_test, labels_test)) + "%"
  + " sur les données de test")

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
cv_train  = np.array([cvv.obsVoisins(i) for i in ims_binarize_train])
cv_test   = np.array([cvv.obsVoisins(i) for i in ims_binarize_test])

class_dis.loss_rate(cv_test, labels_test)
