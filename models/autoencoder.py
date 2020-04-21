import modules
import losses
import activation

class Autoencoder(modules.Composite):
   
    def __init__(self, levels, layers, thickness, n):
        super(Autoencoder, self).__init__()
        self.levels = levels
        self.layers = layers
        self.n = n
        self.thickness = thickness

    def _build(self, X):
        for level in range(self.levels):
            for layer in range(self.layers):
                self.add(f"encoder_conv{level}_{layer}", modules.Conv(self.n, self.thickness*(2**(level + 1)), padding="same"))
                self.add(f"encoder_relu{level}_{layer}", activation.Relu())
            self.add(f"encoder_maxpool{level}", modules.Maxpool(2))
        for level in reversed(range(self.levels)):
            for layer in range(self.layers):
                self.add(f"decoder_conv{level}_{layer}", modules.Conv(self.n, self.thickness*(2**(level + 1)), padding="same"))
            self.add(f"decoder_upsample{level}", modules.Upsample(2))
            self.add(f"decoder_relu{level}_{layer}", activation.Relu())
        self.add("final_conv", modules.Conv(self.n, 3, padding="same"))
        self.add("mse", losses.MSE())

    def _forward(self, X):
        for level in range(self.levels):
            for layer in range(self.layers):
                l = self.get(f"encoder_conv{level}_{layer}")
                X = l(X)
            l = self.get(f"encoder_maxpool{level}")
            X = l(X)
        for level in reversed(range(self.levels)):
            for layer in range(self.layers):
                l = self.get(f"decoder_conv{level}_{layer}")
                X = l(X)
            l = self.get(f"decoder_upsample{level}")
            X = l(X)
        l = self.get("final_conv")
        X = l(X)
        l = self.get("mse")
        return l(X)
            
    def backprop(self, lr, grad):
        l = self.get("mse")
        grad = l.backprop(lr, grad)
        l = self.get("final_conv")
        grad= l.backprop(lr, grad)
        for level in range(self.levels):
            l = self.get(f"decoder_upsample{level}")
            grad = l.backprop(lr, grad)
            for layer in reversed(range(self.layers)):
                l = self.get(f"decoder_conv{level}_{layer}")
                grad = l.backprop(lr, grad)
        for level in reversed(range(self.levels)):
            l = self.get(f"encoder_maxpool{level}")
            grad = l.backprop(lr, grad)
            for layer in reversed(range(self.layers)):
                l = self.get(f"encoder_conv{level}_{layer}")
                grad = l.backprop(lr, grad)

