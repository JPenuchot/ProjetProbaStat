import gzip, pickle, numpy as np
import math
import matplotlib.pyplot as plt

f = gzip.open('./mnist.pkl.gz', 'rb')
u = pickle._Unpickler(f)
u.encoding = 'latin1'
p = u.load()
train_set, valid_set, test_set = p

# Récupération des images et des labels
images = train_set[0]
images_test = test_set[0]
labels = train_set[1]
labels_test = test_set[1]
#Correction professeur

means = np.zeros((10,784))
variances = {}
priors = {}
subtrain = {}

for i in range(0, 10):
    subtrain = images[labels==i]
    mean = subtrain.mean(axis=0)
    means[i] = mean
    var = subtrain.var(axis=0)
    variances[i] = var
    priors[i] = len(images[labels == i])/len(images)

tresh = 0.189
cpt = 0
index = []

ind_pixels = np.argsort(variances[4])
best_pixels_var = ind_pixels[-5:]
print(best_pixels_var[0])
rep_pixels = {}
for i in range(5):
	rep_pixels[i] = np.array([im[best_pixels_var[i]] for im in images])

histo = np.histogram(rep_pixels[0], bins='auto')
plt.hist(np.histogram(rep_pixels[0]), bins=len(rep_pixels[0]))
plt.show()
