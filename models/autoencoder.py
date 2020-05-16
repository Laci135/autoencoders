import modules
import losses
import activation
import numpy as np

# autoencoder module using my deep learning api built solely on np
class Autoencoder(modules.Composite):
   
    # params: number of conv levels. conv layers per level, conv layer thickness, kernel&pool size
    def __init__(self, levels, layers, thickness, n):
        super(Autoencoder, self).__init__()
        self.levels = levels
        self.layers = layers
        self.n = n
        self.thickness = thickness

    # build autoencoder
    # do not call explicitely
    def _build(self, X):
        for level in range(self.levels): # create levelsxlayers conv layers  for the encoder
            for layer in range(self.layers): # add conv layer and relu activation
                self.add(f"encoder_conv{level}_{layer}", modules.Conv(self.n, self.thickness*(2**(level + 1)), padding="same"))
                self.add(f"encoder_relu{level}_{layer}", activation.Relu())
            self.add(f"encoder_maxpool{level}", modules.Maxpool(2)) # add maxpooling
        for level in reversed(range(self.levls)): # now create the decoder, similarly
            for layer in range(self.layers): 
                self.add(f"decoder_conv{level}_{layer}", modules.Conv(self.n, self.thickness*(2**(level + 1)), padding="same"))
                self.add(f"decoder_relu{level}_{layer}", activation.Relu())
            self.add(f"decoder_upsample{level}", modules.Upsample(2)) # we use upsampling
        self.add("final_conv", modules.Conv(self.n, 3, padding="same")) # add final conv layer to reproduce the output image (this keeps locality)
        self.add("mse", losses.MSE()) # add mse loss function

    # fwd pass
    # do not call explicitely
    def _forward(self, X):
        for level in range(self.levels): # pass data through all layers and activations previously created
            for layer in range(self.layers): # for model structure see the _build method
                l = self.get(f"encoder_conv{level}_{layer}")
                X = l(X) 
                print(f"conv -- {np.average(X)}")
                l = self.get(f"encoder_relu{level}_{layer}")
                X = l(X)
                print(f"relu -- {np.average(X)}")
            l = self.get(f"encoder_maxpool{level}")
            X = l(X)
            print(f"maxpool -- {np.average(X)}")
        for level in reversed(range(self.levels)):
            for layer in range(self.layers):
                l = self.get(f"decoder_conv{level}_{layer}")
                X = l(X)
                print(f"conv -- {np.average(X)}")
                l = self.get(f"decoder_relu{level}_{layer}")
                X = l(X)
                print(f"relu -- {np.average(X)}")
            l = self.get(f"decoder_upsample{level}")
            X = l(X)
            print(f"upsample -- {np.average(X)}")
        l = self.get("final_conv")
        X = l(X)
        print(f"final conv -- {np.average(X)}")
        l = self.get("mse")
        return l(X)
            
    def backprop(self, grad): # pass backprop data backwards through all layers and losses, in reverse order
        l = self.get("mse")
        grad = l.backprop(grad)
        print(f"mse grad  -- {np.average(grad)}")
        l = self.get("final_conv")
        grad = l.backprop(grad)
        print(f"final conv grad  -- {np.average(grad)}")
        for level in range(self.levels):
            l = self.get(f"decoder_upsample{level}")
            grad = l.backprop(grad)
            print(f"upsample grad  -- {np.average(grad)}")
            for layer in reversed(range(self.layers)):
                l = self.get(f"decoder_relu{level}_{layer}")
                grad = l.backprop(grad)
                print(f"relu grad  -- {np.average(grad)}")
                l = self.get(f"decoder_conv{level}_{layer}")
                grad = l.backprop(grad)
                print(f"conv grad  -- {np.average(grad)}")
        for level in reversed(range(self.levels)):
            l = self.get(f"encoder_maxpool{level}")
            grad = l.backprop(grad)
            print(f"maxpool grad  -- {np.average(grad)}")
            for layer in reversed(range(self.layers)):
                l = self.get(f"encoder_relu{level}_{layer}")
                grad = l.backprop(grad)
                print(f"relu grad  -- {np.average(grad)}")
                l = self.get(f"encoder_conv{level}_{layer}")
                grad = l.backprop(grad)
                print(f"conv grad  -- {np.average(grad)}")


