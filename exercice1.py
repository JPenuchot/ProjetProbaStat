import numpy as np

  def __init__(self, nm_classes, images, labels):
    self.nm_classes = nm_classes

    for i in range(nm_classes):
      subtrain            = images[labels==i]
      self.means      [i] = subtrain.mean(axis=0)
      self.variances  [i] = subtrain.var(axis=0)
      self.priors     [i] = len(images[labels == i]) / len(images)

def covVoisins(self, images):
	v_g = self.means.roll(1)
	v_d = self.means.roll(-1)

