# Version python3 pour le chargement des données
import pickle, gzip, numpy as np
import bayesien_naif_discret as pps

# Load the dataset
f = gzip.open('./mnist.pkl.gz', 'rb')
u = pickle._Unpickler(f)
u.encoding = 'latin1'
p = u.load()
train_set, valid_set, test_set = p

# Bayesien.py

# Récupération des images et des labels
images      = train_set[0]
images_test = test_set[0]

labels      = train_set[1]
labels_test = test_set[1]

class_con = pps.bn(10, images, labels)

print("L'erreur est de "
  + str(class_con.loss_rate(images_test, labels_test)) + "%"
  + " sur les " + str(len(images)) + " données de test")

# Naïf discret


ims_binarize_train  = np.array(pps.binarize(images))
ims_binarize_test   = np.array(pps.binarize(images_test))

class_dis = pps.bn(10, ims_binarize_train, labels)

alpha = 0.00001

print("L'erreur est de "
  + str(class_dis.loss_rate(ims_binarize_test, labels_test)) + "%"
  + " sur les données de test")
