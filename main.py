# Version python3 pour le chargement des données
import pickle, gzip, numpy as np
import matplotlib.pyplot as plt
import matplotlib

import bayesien_naif_discret as bnd

# Load the dataset
f = gzip.open('./mnist.pkl.gz', 'rb')
u = pickle._Unpickler(f)
u.encoding = 'latin1'
p = u.load()
train_set, valid_set, test_set = p

######################################
# exemple de manipulation

im = train_set[0][0] # the first image
label = train_set[1][0] # its label

print(type(im))
print(im.shape)

# Visualisation d'une image:
im = train_set[0][0]
plt.imshow(im.reshape(28,28) , plt.cm.gray)
plt.show()

# Bayesien.py

# Récupération des images et des labels
images = train_set[0]
images_test = test_set[0]

labels = train_set[1]
labels_test = test_set[1]



print("L'erreur est de " + str(bnd.loss_rate(images_test, labels_test))+ "%"
  + " sur les données de test")

# Naïf discret

print(len(images))
ims_binarize_train = np.array(bnd.binarize(images))
ims_binarize_test = np.array(bnd.binarize(images_test))
alpha = 0.00001

print("L'erreur est de " + str(bnd.loss_rate(ims_binarize_test, labels_test)) + "%"
  + " sur les données de test")
