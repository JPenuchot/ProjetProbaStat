import numpy as np
import random
import math

class mg:
  al = [{} , {}]
  mn = [{} , {}]
  vr = [{} , {}]
  pr = {}

  def __init__(self, images, labels):
    self.classes = np.unique(labels)
    self.pr = [(labels == c).size / labels.size for c in self.classes]

    for c in self.classes:
      subtrain = images[labels == c]
      mn = np.mean(subtrain, axis = 0)
      vr = np.var(subtrain, axis = 0)

      self.mn[0][c], self.mn[1][c] = mn, mn
      self.vr[0][c], self.vr[1][c] = vr, vr

      self.al[0][c] = random.random()
      self.al[1][c] = 1. - self.al[0][c]

  def contribImage(self, image, c, a):
    # Récupération des means, sigma etc.
    means   = self.mn[a][c]
    sigma2  = self.vr[a][c]
    prior   = self.pr[c]

    # Calcul de la contribution
    nonNull = sigma2 > 0
    scale   = 0.5 * np.log(2 * sigma2[nonNull] * math.pi)
    expterm = - 0.5 * np.divide ( np.square(image[nonNull] - means[nonNull])
                                , sigma2[nonNull] )
    llh     = (expterm - scale) #.sum()
    post    = llh + np.log(prior) # À tester
    return post

  def contribImages(self, images, c, a):
    sum = np.zeros(images[0].shape)
    for img in images:
      sum += self.contribImage(img, c, a)
    return sum / images.size

  def fitClass(self, c, images, labels):
    subtrain = images[labels == c]

  #def fit(self, images, labels):
  #  for c in self.classes:

