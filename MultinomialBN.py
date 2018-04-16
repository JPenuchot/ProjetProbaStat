import numpy as np
import math

class MultinomialBN():

	def __init__(self, images_train, labels_train, images_test, labels_test, alpha):

		self.alpha = alpha
		self.images_train = images_train
		self.labels_train = labels_train
		self.images_test = images_test
		self.labels_test = labels_test
		self.nb_images = self.images_train.shape[0]
		self.subtrain = [[im for im, lbl in zip(self.images_train, self.labels_train) if lbl == classe] for classe in range(10)]
		self.log_priors = [np.log(len(c) / self.nb_images) for c in self.subtrain]
		self.count_active_pixels = np.array([np.array(i).sum(axis=0) for i in self.subtrain]) + alpha
		self.np_image_per_class = np.array([len(i) + 2*alpha for i in self.subtrain])
		self.prob_of_features = self.count_active_pixels / self.np_image_per_class[np.newaxis].T

	def computePosteriors3(self):

		calc = [(np.log(self.prob_of_features) * im + np.log(1 - self.prob_of_features) * np.abs(im - 1)).sum(axis=1) + self.log_priors for im in self.images_test]
		list_of_responses = np.argmax(calc, axis = 1)
		return list_of_responses

	def error_rate(self):

	    error_rate = 0
	    res = self.computePosteriors3()
	    for i, res_s in enumerate(res):
	        if res_s != self.labels_test[i]:
	            error_rate += 1
	    return (error_rate / len(self.images_test) * 100).__round__(3)