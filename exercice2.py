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
pixels = []
for p in variances[5]:
	if (cpt < 5 and p > tresh):
		pixels.append(p)
		cpt += 1

print(pixels)

plt.hist(pixels, 50, normed = 1, facecolor='g', alpha = 0.75)
plt.axis([1, 5, 0, 0.3])

plt.show()