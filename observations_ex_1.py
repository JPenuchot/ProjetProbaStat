import pickle, gzip, numpy as np
import matplotlib.pyplot as plt

# Import des fonctions covVoisins etc.
import bayesien_naif as pps
import covvoisins as cvv

# Load the dataset
f = gzip.open('./mnist.pkl.gz', 'rb')
u = pickle._Unpickler(f)
u.encoding = 'latin1'
p = u.load()
train_set, valid_set, test_set = p

# Récupération des images et des labels
images      = train_set[0]
images_test = test_set[0]

labels      = train_set[1]
labels_test = test_set[1]

print()
print("# Observation des covariances des pixels de la classe 5:")
print()

class5 = images[labels == 5]
plt.imshow(np.cov(class5, rowvar = False))
plt.show()

print()
print("# Observation des covariances des pixels de la classe 9:")
print()

class9 = images[labels == 9]
plt.imshow(np.cov(class9, rowvar = False))
plt.show()

print()
print("# Observation des covariances entre voisins de la classe 5:")
print()

# On entraîne un modèle par flemme de refaire toutes les moyennes etc
class_con = pps.bn(10, images, labels)
var_voisins = cvv.covVoisins(class5, class_con.means[5], class_con.variances[5])

# var_voisins contient dans l'ordre :
#
# variance,
# covariance avec voisins de gauche,
# covariance avec voisins de droite.

plt.imshow(var_voisins[784:784*2].reshape(28, 28))
plt.show()
