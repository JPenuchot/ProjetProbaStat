import numpy as np

  def __init__(self, nm_classes, images, labels):
    self.nm_classes = nm_classes

    for i in range(nm_classes):
      subtrain            = images[labels==i]
      self.means      [i] = subtrain.mean(axis=0)
      self.variances  [i] = subtrain.var(axis=0)
      self.priors     [i] = len(images[labels == i]) / len(images)

def covVoisins(self, images):
	# Décalage des variances pour observer les variances
  # des pixels par rapport à leurs voisins
  E_g = np.roll(self.means, 1, axis = 1)
	E_d = np.roll(self.means, -1, axis = 1)

  # Décalage des images pour observer les pixels
  # par rapport à leurs voisins
  images_g = np.roll(images, 1, axis = 1)
  images_d = np.roll(images, -1, axis = 1)

  # Produit et somme...
  prob_xg = np.prod(images_g, axis = 1).sum(axis = 1) / images.size
  prob_xd = np.prod(images_d, axis = 1).sum(axis = 1) / images.size

  # Calcul des covariances aux voisins...
  res_g = prob_xg - (E * E_g)
  res_d = prob_xd - (E * E_d)

  # Résultat
  return res_g, res_d

def obsVoisins(self, image):
  image_g = image.roll( 1)
  image_d = image.roll(-1)

  res_g = image * image_g
  res_d = image * image_d

  return res_g, res_d
