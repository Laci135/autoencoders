> Noise filtering (convolutional and FC, SVHN dataset) and VAE (convolutional, KMNIST, MNIST) have been added to the keras directory.

# autoencoders
Neural networks homework BME-MIT 2020 spring

I have tried what happens if I implement an autoencoder using pure numpy. Sadly performance was poor so I switched to keras...

SVHN dataset: http://ufldl.stanford.edu/housenumbers/
MNIST dataset: http://yann.lecun.com/exdb/mnist/
KMNIST dataset: https://github.com/rois-codh/kmnist

###Running
Pure numpy implementation:
`python3 __main__.py`

The notebooks in the keras directory may be run in any Jupyter environment.

Sample output of the pure numpy implementation -- SVHN dataset -- 20s/batch, ~6k batches/epoch :

epoch 0 started\
conv -- 113.26181821469908\
relu -- 113.26181821469908\
conv -- 110.170841169946\
relu -- 110.170841169946\
maxpool -- 117.87560281635804\
conv -- 109.60671858924898\
relu -- 109.60671858924898\
conv -- 103.88907573715261\
relu -- 103.88907573715261\
maxpool -- 116.03859532972616\
conv -- 100.09674010144118\
relu -- 100.09674010144118\
conv -- 89.13909991350953\
relu -- 89.13909991350953\
upsample -- 89.13909991350954\
conv -- 84.6542538240293\
relu -- 84.6542538240293\
conv -- 81.72870531566652\
relu -- 81.72870531566652\
upsample -- 81.72870531566652\
final conv -- 80.36860573078202\
mse grad  -- -8.036860573078202e-07\
final conv grad  -- -2.648940547985907e-07\
upsample grad  -- -2.648940547985907e-07\
relu grad  -- -2.648940547985907e-07\
conv grad  -- -1.1420030958689994e-07\
relu grad  -- -1.1420030958689994e-07\
conv grad  -- -4.9247142726086446e-08\
upsample grad  -- -4.9247142726086446e-08\
relu grad  -- -4.9247142726086446e-08\
conv grad  -- -4.017550595439327e-08\
relu grad  -- -4.017550595439327e-08\
conv grad  -- -3.269161512513887e-08\
maxpool grad  -- -3.2691615125138866e-08\
relu grad  -- -3.2691615125138866e-08\
conv grad  -- -2.9059213444567887e-08\
relu grad  -- -2.9059213444567887e-08\
conv grad  -- -2.5105499986306203e-08\
maxpool grad  -- -2.5105499986306203e-08\
relu grad  -- -2.5105499986306203e-08\
conv grad  -- -1.115799999391387e-08\
relu grad  -- -1.115799999391387e-08\
conv grad  -- -4.921991485085247e-09\
epoch 0 -- 0/6105 -- loss: 169712875.44767815\
conv -- 123.05982463998367\
relu -- 123.05982463998367\
conv -- 119.64422084808761\
relu -- 119.64422084808761\
maxpool -- 128.8495395391595\
conv -- 119.72278843973413\
relu -- 119.72278843973413\
conv -- 113.41730488403941\
relu -- 113.41730488403941\
maxpool -- 127.22930470101191\
conv -- 109.68025653857842\
relu -- 109.68025653857842\
conv -- 97.65195761662831\
relu -- 97.65195761662831\
upsample -- 97.65195761662831\
conv -- 92.72921726519122\
relu -- 92.72921726519122\
conv -- 89.51558546809565\
relu -- 89.51558546809565\
upsample -- 89.51558546809565\
final conv -- 88.01281952964904\
mse grad  -- -8.801281952964903e-07\
final conv grad  -- -2.9004848502392735e-07\
upsample grad  -- -2.900484850239274e-07\
relu grad  -- -2.900484850239274e-07\
conv grad  -- -1.2503427730837768e-07\
relu grad  -- -1.2503427730837768e-07\
conv grad  -- -5.3916013025416374e-08\
upsample grad  -- -5.3916013025416374e-08\
relu grad  -- -5.3916013025416374e-08\
conv grad  -- -4.397975740656026e-08\
relu grad  -- -4.397975740656026e-08\
conv grad  -- -3.578431472483425e-08\
maxpool grad  -- -3.578431472483425e-08

...
