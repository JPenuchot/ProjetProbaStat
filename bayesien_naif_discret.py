import numpy as np

# Correction professeur

# Binarisation pour le bayesien naïf discret
def binarize(im_test):
  tresh = 0.5
  res   = []

  for i in range(len(im_test)):
    curr_res = np.zeros(im_test[i].shape)
    curr_res[im_test[i] >= tresh] = 1
    res.append(curr_res)

  return res

## Bayesien naïf
class bn:

  nm_classes = 0
  means = {}
  variances = {}
  priors = {}
  sum_y = {}

  __init__(self, nm_classes, means, variances, priors, sum_y):
    for i in range(nm_classes):
      subtrain = images[labels==i]
      sum_y[i] = subtrain.sum(axis = 1)

      print(sum_y)

      mean          = subtrain.mean(axis=0)
      means[i]      = mean
      var           = subtrain.var(axis=0)
      variances[i]  = var
      priors[i]     = len(images[labels == i])/len(images)

  def computePosteriors2(image):
    posteriors = np.zeros([nb_classes,1])

    for lbl in range(nm_classes):
      mean      = means[lbl]
      sigma2    = variances[lbl]
      non_null  = sigma2 != 0
      scale     =  0.5 * np.log(2 * sigma2[non_null] * math.pi)
      expterm   = -0.5 * np.divide( np.square(image[non_null] - mean[non_null])
                                  , sigma2[non_null] )
      llh   = (expterm - scale).sum()
      post  = llh + np.log(priors[lbl])
      posteriors[lbl] = post
    return posteriors

  def loss_rate(im_test,labs_test):
    res = 0

    for i in range(len(im_test)):
      l = list(computePosteriors2(im_test[i]))
      res += 1 if labels_test[i] != l.index(max(l)) else 0

    return res / len(im_test) * 100


## Bayesien naïf discret
class bnd:

  nm_classes = 0
  means = {}
  variances = {}
  priors = {}
  sum_y = {}

  __init__(self, nm_classes, means, variances, priors, sum_y):
    images = binarize(images)
    for i in range(nb_classes):
      subtrain  = images[labels==i]
      sum_y[i]  = subtrain.sum(axis = 1)

      print(sum_y)

      mean          = subtrain.mean(axis=0)
      means[i]      = mean
      var           = subtrain.var(axis=0)
      variances[i]  = var
      priors[i]     = len(images[labels == i])/len(images)

  def computePosteriors2(image):
    # Binarisation (Bayesien Discret)
    image = binarize(image)
    posteriors = np.zeros([nb_classes,1])

    for lbl in range(nm_classes):
      mean      = means[lbl]
      sigma2    = variances[lbl]
      non_null  = sigma2 != 0
      scale     =   0.5 * np.log(2 * sigma2[non_null] * math.pi)
      expterm   = - 0.5 * np.divide( np.square(image[non_null] - mean[non_null])
                                   , sigma2[non_null] )
      llh       = (expterm - scale).sum()
      post      = llh + np.log(priors[lbl])
      posteriors[lbl] = post
    return posteriors

  def loss_rate(im_test,labs_test):
    im_test = binarize(im_test)
    res     = 0

    for i in range(len(im_test)):
      l = list(computePosteriors2(im_test[i]))
      res += 1 if labels_test[i] != l.index(max(l)) else 0

    return res / len(im_test) * 100
