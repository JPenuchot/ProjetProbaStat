# Version python3 pour le chargement des données
import pickle, gzip, numpy as np
import matplotlib.pyplot as plt
import matplotlib

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

# Correction professeur
means = {}
variances = {}
priors = {}
sum_y = {}

for i in range(0, 10):
  subtrain = images[labels==i]
  sum_y[i] = subtrain.sum(axis = 1)
  print(sum_y)
  mean = subtrain.mean(axis=0)
  means[i] = mean
  var = subtrain.var(axis=0)
  variances[i] = var
  priors[i] = len(images[labels == i])/len(images)

print("l'erreur est de " + str(loss_rate(images_test, labels_test))+ "%"
  + " sur les données de test")

# Naïf discret

print(len(images))
ims_binarize_train = np.array(binarize(images))
ims_binarize_test = np.array(binarize(images_test))
alpha = 0.00001

print("l'erreur est de " + str(loss_rate(ims_binarize_test, labels_test)) + "%"
  + " sur les données de test")
