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

TEST


## Ideas

- image retrieval using encoding + LSH
- latent codes comparison
- t-sne visualisierung
- denoising examples


## Concept

Autoencoders are structured to take an input, transform this input into a different representation, an _embedding_



## Math

### Vanilla Autoencoder

<!-- $$
\begin{align*}
  & \phi(x,y) = \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right)
  = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j) = \\
  & (x_1, \ldots, x_n) \left( \begin{array}{ccc}
      \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\
      \vdots & \ddots & \vdots \\
      \phi(e_n, e_1) & \cdots & \phi(e_n, e_n)
    \end{array} \right)
  \left( \begin{array}{c}
      y_1 \\
      \vdots \\
      y_n
    \end{array} \right)
\end{align*}
$$ -->


$$
\begin{align*}
f(g(x)) = 0
\end{align*}
$$

### Sparse Autoencoder

The term _sparse_ autoencoder entails a whole type of autoencoders, of which I will talk about later. Sparse autoencoder enforce sparsity on the latent variables z after encoding the input. This means that 

$$
\begin{align*}
loss = x - f(g(x)) ^2
\end{align*}
$$

### Denoising Autoencoder

$$
\begin{align*}
loss = x - f(g(x)) ^2
\end{align*}
$$


### Contractive Autoencoder

$$
\begin{align*}
loss = ||x - f(g(x))||^2
\end{align*}
$$

# Code


 {% raw %}
  $$a^2 + b^2 = c^2$$ --> note that all equations between these tags will not need escaping! 
 {% endraw %}




# Experiments

## Neighborhood search with Local Sensitivity Hashing

LSH uses a locality-preserving hash function that maps a point in a multidimensional space to a scalar value. Each feature vector is binned into a specific hash value based on its location in feature space. Feature vectors close in feature space have a high probability of being hashed into the same bin. After the initial build of the hashing function which has to be done only once after the initial hashing 


LSH reduces the runtime complexity of neighborhood from $$\mathcal{O}(N^2)$$ to $$\mathcal{O}(N \log N)$$ where $$N$$ denotes the number of feature vectors. Even though LSH does not guarantee finding the closest neighbor, it finds it with very high probability which is sufficient for our purpose.
