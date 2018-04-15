import pickle, gzip, numpy as np
import matplotlib.pyplot as plt

# Import des fonctions covVoisins etc.
import bayesien_naif_discret as pps
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

classe4 = images[labels == 4]

variances = np.var(classe4, axis = 0)
max_ids   = np.argsort(variances)

for i in max_ids[-5:]:
  # Afficher l'histogramme du pixel de rang i
  val_px = np.array([img[i] for img in classe4])
  plt.hist(np.histogram(val_px, bins=np.arange(np.argmax(val_px))))
  plt.show()
