import numpy as np

  def __init__(self, nm_classes, images, labels):
    self.nm_classes = nm_classes

    for i in range(nm_classes):
      subtrain            = images[labels==i]
      self.means      [i] = subtrain.mean(axis=0)
      self.variances  [i] = subtrain.var(axis=0)
      self.priors     [i] = len(images[labels == i]) / len(images)

def covVoisins(self, subtrain, means):
  # Décalage des images pour observer les pixels
  # par rapport à leurs voisins
  subtrain_g = subtrain.roll( 1, axis = 1)
  subtrain_d = subtrain.roll(-1, axis = 1)

  # Produit et somme...
  prob_xg = np.multiply(subtrain, subtrain_g).sum(axis = 0) / subtrain.size
  prob_xd = np.multiply(subtrain, subtrain_d).sum(axis = 0) / subtrain.size

  # Décalage des variances pour observer les variances
  # des pixels par rapport à leurs voisins
  E_g = self.means.roll( 1, axis = 1)
  E_d = self.means.roll(-1, axis = 1)

  # Calcul des covariances aux voisins...
  res_g = prob_xg - (E * E_g)
  res_d = prob_xd - (E * E_d)

  # Résultat
  return np.concatenate((res_g, res_d))

def obsVoisins_image(self, image):
  image_g = image.roll( 1)
  image_d = image.roll(-1)

  res_g = image * image_g
  res_d = image * image_d

  return np.concatenate((res_g, res_d))
