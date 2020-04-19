import modules.module
import modules.conv

class Autoencoder(modules.Composite):
   
    def __init__(self, levels, layers, thickness, n):
        self. levels = levels
        self.layers = layers
        self.n = n

    def _build(self, X):
        for level in range(levels):
            for layer in range(layers):
                add(f"encoder_conv{level}_{layer}", Conv(n, thickness*(2**level + 1), stride=1, padding="same"))
            add(f"encoder_maxpool{level}", Maxpool(n))
            for layer in reversed(range(layers)):
                add(f"decoder_conv{level}_{layer}", Conv(n, thickness*(2**level - 1), stride=1, padding="same"))
            add(f"decoder_upsample{level}", Upsample(n))
        add("mse", MSE())

    def _forward(self, X):
        for level in range(levels):
            for layer in range(layers):
                l = get(f"encoder_conv{level}_{layer}")
                X = l(X)
            l = get(f"encoder_maxpool{level}", Maxpool(n))
            X = l(X)
        for level in reversed(range(levels)):
            for layer in range(layers):
                l = get(f"decoder_conv{level}_{layer}")
                X = l(X)
            l = get(f"decoder_upsample{level}")
            X = l(X)
        l = get("mse")
        return l(X)
            
    def backprop(self, lr, loss):
        l = get("mse")
        loss = l.backprop(lr)
        for level in range(levels):
            l = get(f"decoder_upsample{level}")
            loss = l.backprop(loss)
            for layer in reversed(range(layers)):
                l = get(f"decoder_conv{level}_{layer}")
                loss = l.backprop(loss)
        for level in reversed(range(levels)):
            l = get(f"encoder_maxpool{level}")
            loss = l.backprop(loss)
            for layer in reversed(range(layers)):
                l = get(f"encoder_conv{level}_{layer}")
                loss = l.backprop(loss)

