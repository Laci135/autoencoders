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
                layer = get(f"encoder_conv{level}_{layer}")
                X = layer(X)
            layer = get(f"encoder_conv{level}_maxpool", Maxpool(n))
            X = layer(X)
            for layer in reversed(range(layers)):
                layer = get(f"decoder_conv{level}_{layer}")
                X = layer(X)
            layer = get(f"decoder_upsample{level}")
            X = layer(X)
        layer = get("mse")
        return layer(X)
            
    def backprop(self, lr, loss):
        
