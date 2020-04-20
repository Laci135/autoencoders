import models
import numpy as np
import math
import scipy.io
import matplotlib.pyplot as plt

images = np.rollaxis(np.array(scipy.io.loadmat("SVHN.mat")['X']), axis = 3, start=0)

noise = np.random.binomial(300, 0.5, images.shape)-150

#noisy_images = images + noise

model = models.Autoencoder(levels = 2, layers = 2, thickness = 3, n = 3)

batch_size = 12

for b in range(math.ceil(images.shape[0] / batch_size)):
    #X = noisy_images[batch_size*b:batch_size*(b+1)]
    Y_target = images[batch_size*b:batch_size*(b+1)]
    model(Y_target)
    #Y = model(X)
