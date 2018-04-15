import numpy as np

def covVoisins(subtrain, means, cov):
  # Décalage des images pour observer les pixels
  # par rapport à leurs voisins
  subtrain_g = np.roll(subtrain,  1, axis = 1)
  subtrain_d = np.roll(subtrain, -1, axis = 1)

  # Produit et somme...
  prob_xg = np.multiply(subtrain, subtrain_g).sum(axis = 0) / subtrain.size
  prob_xd = np.multiply(subtrain, subtrain_d).sum(axis = 0) / subtrain.size

  # Décalage des variances pour observer les variances
  # des pixels par rapport à leurs voisins
  E_g = np.roll(means,  1)
  E_d = np.roll(means, -1)

  # Calcul des covariances aux voisins...
  res_g = prob_xg - (means * E_g)
  res_d = prob_xd - (means * E_d)

  # Résultat
  return np.concatenate((cov, res_g, res_d))

def obsVoisins(image):
  image_g = np.roll(image,  1)
  image_d = np.roll(image, -1)

  res_g = image * image_g
  res_d = image * image_d

  return np.concatenate((image, res_g, res_d))

def meanVoisins(subtrain):
  res = np.array([obsVoisins(i) for i in subtrain])
  return res.mean(axis = 0)
