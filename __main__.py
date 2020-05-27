import models # pkg of self-made network models
import numpy as np # numpy for working with matrices
import math # math for mathematic operations
import scipy.io # scipy for accessing data
import matplotlib.pyplot as plt # pyplot for plotting

# load and reshape images
images = np.rollaxis(np.array(scipy.io.loadmat("datasets/SVHN.mat")['X']), axis = 3, start=0)

# generate binomial noise
noise = np.random.binomial(300, 0.5, images.shape)-150

# add noise
noisy_images = images + noise

# instantiate autoencoder
model = models.Autoencoder(levels = 2, layers = 2, thickness = 2, n = 3)

epochs = 100 # num of epochs
batch_size = 12 # batch-size
lr = 0.00000001 # learning rate

batches = math.ceil(images.shape[0] / batch_size) # calculate batch count

for e in range(epochs):
    print (f"epoch {e} started")
    for b in range(batches): # iterate over all batches
        X = noisy_images[batch_size*b:batch_size*(b+1)] # create inp data and target batches
        Y_target = images[batch_size*b:batch_size*(b+1)]
        Y = model(X) # fwd pass

        loss = model.get("mse").get(Y_target) # calculate loss matrix
        loss_total = model.get("mse").total(Y_target) # calculate loss total
        gradient = model.get("mse").get_grad(Y_target) # calculate grad mx form loss mx

        
        model.backprop(gradient * lr) # perform backprop
        print(f"epoch {e} -- {b}/{batches} -- loss: {loss_total}")

    print(f"epoch {e} done: ")
