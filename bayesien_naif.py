import numpy as np
import math

# Correction professeur

# Binarisation pour le bayesien naïf discret
def binarize(im_test):
  tresh = 0.5
  res = []
  for i in range(len(im_test)):
    curr_res = np.zeros(im_test[i].shape)
    curr_res[im_test[i] >= tresh] = 1
    res.append(curr_res)
  return res

## Bayesien naïf
class bn:
  nm_classes  = 0
  means       = {}
  variances   = {}
  priors      = {}

  def __init__(self, nm_classes, images, labels):
    self.nm_classes = nm_classes

    for i in range(self.nm_classes):
      subtrain            = images[labels == i]

      self.means      [i] = subtrain.mean(axis=0)
      self.variances  [i] = subtrain.var(axis=0)
      self.priors     [i] = len(images[labels == i]) / len(images)

  def computePosteriors2(self, image):
    posteriors = np.zeros([self.nm_classes, 1])

    for lbl in range(self.nm_classes):
      mean      = self.means[lbl]
      sigma2    = self.variances[lbl]
      non_null  = sigma2 > 0

      scale     =   0.5 * np.log(2 * sigma2[non_null] * math.pi)

      expterm   = - 0.5 * np.divide( np.square(image[non_null] - mean[non_null])
                                   , sigma2[non_null] )

      llh   = (expterm - scale).sum()
      post  = llh + np.log(self.priors[lbl])
      posteriors[lbl] = post
    return posteriors

  def loss_rate(self, im_test, labs_test):
    res = 0

    for i in range(len(im_test)):
      l = list(self.computePosteriors2(im_test[i]))
      res += 1 if labs_test[i] != l.index(max(l)) else 0

    return float(res) / len(im_test) * 100
