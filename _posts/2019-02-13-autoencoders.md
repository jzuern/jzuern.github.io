---
layout: post
title: "But what are autoencoders?"
description: "A high-level introduction into autoencoders"
date: 2019-02-15
tags: blog
comments: true
use_math: true
---


In todays post I would like to give you a quick-and-dirty introduction into one of the mo


## Ideas

- image retrieval using encoding + LSH
- latent codes comparison
- t-sne visualisierung
- denoising examples


## Concept

Autoencoders are structured to take an input, transform this input into a different representation, an _embedding_ of the input. From this input, it aims to reconstruct the input as precicely as possible. This mapping from input to reconstructed input can obviously not be exact if the embedding contains less information than the input. If this is the case, the autoencoder is called *undercomplete*. In this post, we will exclusively deal with undercomplete autoencoders. But why would someone actually want to have an undercomplete autoencoder? Using a low-dimensional embedding layer forces the autoencoder during training to find an embedding or _code_ (hence the name autoencoder) that somewhat represents the input in a concrete way but contains far less bits. It is a form of _Dimensionality reduction_.



## Math




We employ a convolutional autoencoder to obtain a compact encoding of patch content in the image feature space. We used the Deep VAE from the paper 'Deep Embedded Clustering with Data Augmentation' to find feature-space embeddings for each patch. The approach consists of two independent ideas:

A convolutional autoencoder is used to generate low-dimensional representations for the images of patches. Our bottleneck layer has 32 dimensions. During training of the autoencoder, the input image resolutions are reduced while the number of image channels is increased using convolutional filters. The  image is flattened and using a Fully Connected Layer reduced to the 32-dimensional feature vector in latent space. This is the so-called encoder of the network since image information is encoded (== boiled down) in the feature vector. In the subsequent decoder, the image is reconstructed from the feature vector and scaled up using transposed convolutional layers.

More formally, an autoencoder is a nonlinear mapping of an input $x$ into an output $$x$$ using an intermediate representation $$x_{encoded} = f_{encode}(x)$$. During training the encoding part of the neural network learns a nonlinear mapping of $x$ into $$x_{encoded}$$. The decoder, on the other hand, learns a nonlinear mapping from $$x_{encoded}$$ into the original space. The goal of training is to minimize a loss 





### Vanilla Autoencoder

The most basic example of an autoencoder can be defined as follows:

$$
\begin{align*}
loss = \mathcal{L}(f(g(\mathbf{x})) - \mathbf{x}) \rightarrow Min
\end{align*}
$$

The term $$\mathcal{L}$$ stands for an arbitrary loss function. An exemplary loss function typically encountered with autoencoders is the MSE (mean squared error) between autoencoder input and output: 

$$\mathcal{L}(\mathbf{x};\mathbf{\theta})=\|\mathbf{x}-f(g(\mathbf{x}))\|)^2 $$.


If we set all activation functions of the encoder to linear functions (no ReLu, tanh or similar), it can be shown that the encoder learns in fact to perform PCA (Principal Component Analysis)! The intuition behind this fact is that the principal components of an arbitrary distribution within the input $$x$$ is




### Convolutional Autoencoder

A convolutional autoencoder has layer-types in its encoder and decoder that _convolve_ the input in some way. Typical examples for these kinds of layers are 2-D convolutions or transposed 2-D convolutions.




### Sparse Autoencoder

The term _sparse_ autoencoder entails a whole type of autoencoders, of which I will talk about later. Sparse autoencoder enforce sparsity on the latent variables z after encoding the input. This means that 

$$
\begin{align*}
loss = \mathcal{L}(f(g(\mathbf{x})) - \mathbf{x}) + \Omega(\mathbf{h})
\end{align*}
$$



### Denoising Autoencoder

Loss not between ---

$$
\begin{align*}
loss = \mathbf{x} - f(g(\tilde{\mathbf{x}})) ^2
\end{align*}
$$


### Contractive Autoencoder

$$
\begin{align*}
\Omega(\mathbf{h}) = \lambda \sum_i  ||\nabla_x h_i||^2
\end{align*}
$$

This lets extraction function resist infinitesimal permutations w.r.t. the input $$\mathbf{x}$$

# Code


```python
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

# serialize model to JSON
model_json = autoencoder.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("model.h5")
print("Saved model to disk")
```

# Experiments

## Neighborhood search with Local Sensitivity Hashing

LSH uses a locality-preserving hash function that maps a point in a multidimensional space to a scalar value. Each feature vector is binned into a specific hash value based on its location in feature space. Feature vectors close in feature space have a high probability of being hashed into the same bin. After the initial build of the hashing function which has to be done only once after the initial hashing 


LSH reduces the runtime complexity of neighborhood from $$\mathcal{O}(N^2)$$ to $$\mathcal{O}(N \log N)$$ where $$N$$ denotes the number of feature vectors. Even though LSH does not guarantee finding the closest neighbor, it finds it with very high probability which is sufficient for our purpose.
