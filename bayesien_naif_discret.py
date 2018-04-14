import numpy as np

# Binarisation pour le bayesien naïf discret
def binarize(im_test):
  tresh = 0.5
  res   = []
  for i in range(len(im_test)):
    curr_res = np.zeros(im_test[i].shape)
    curr_res[im_test[i] >= tresh] = 1
    res.append(curr_res)
  return res

def computePosteriors2(image):
  posteriors = np.zeros([10,1])
  for lbl in range(10):
    mean      = means[lbl]
    sigma2    = variances[lbl]
    non_null  = sigma2 != 0

    scale     =  0.5 * np.log(2 * sigma2[non_null] * math.pi)
    expterm   = -0.5 * np.divide( np.square(image[non_null]-mean[non_null])
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
