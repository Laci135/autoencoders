import models
import numpy as np
import math
import scipy.io
import matplotlib.pyplot as plt

images = np.rollaxis(np.array(scipy.io.loadmat("datasets/SVHN.mat")['X']), axis = 3, start=0)

noise = np.random.binomial(300, 0.5, images.shape)-150

noisy_images = images + noise

model = models.Autoencoder(levels = 2, layers = 2, thickness = 2, n = 3)

epochs = 100
batch_size = 12
lr = 0.00000001

batches = math.ceil(images.shape[0] / batch_size)
for e in range(epochs):
    print (f"epoch {e} started")
    for b in range(batches):
        X = noisy_images[batch_size*b:batch_size*(b+1)]
        Y_target = images[batch_size*b:batch_size*(b+1)]
        Y = model(X)

        loss = model.get("mse").get(Y_target)
        loss_total = model.get("mse").total(Y_target)
        gradient = model.get("mse").get_grad(Y_target)

        
        model.backprop(gradient * lr)
        print(f"epoch {e} -- {b}/{batches} -- loss: {loss_total}")

    print(f"epoch {e} done: ")
