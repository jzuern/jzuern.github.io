---
layout: post
title: "What's new with TensorFlow 2.0?"
description: "A guide through the past and the future of TensorFlow"
date: 2019-04-05
tags: blog
comments: true
use_math: true
---


| ![](/images/tf2.0/3.png) | 
|:--:| 
| *TensorFlow logo* |




The machine learning library TensorFlow has had a long history of releases starting from the initial open-source release from the Google Brain team back in November 2015. Initially developed internally under the name DistBelief, TensorFlow quickly rose to become the most widely used machine learning library today. And not without reason.


## TensorFlow 1.XX — where are we today?



| ![](/images/tf2.0/1.png) | 
|:--:| 
| *GitHub repository stars over time for the most widely used machine learning libraries* |



Before we discuss the most important changes for TensorFlow 2.0, let us quickly recap the some of the essential aspects of TensorFlow 1.XX:

### Language support

Python was the first client language supported by TensorFlow and currently supports the most features within the TensorFlow ecosystem. Nowadays, TensorFlow is available in a multitude of programming languages. The TensorFlow core is written in pure C++ for better performance and is exposed via a C API. Apart from the bindings to Python2.7/3.4–3.7, TensorFlow also offers support for JavaScript ([Tensorflow.js](https://www.tensorflow.org/js)), Rust and R. Especially the syntactically simple Python API, compared to the brittle explicitness of C/C++ allowed TensorFlow to quickly overtake the Caffe machine learning library, an early-day competitor.

### Computation Graph

From the start, the core of TensorFlow has been the so-called Computation Graph. In this graph model, each operation (Add, Multiply, Subtract, Logarithmize, Matrix-Vector Algebra, Complex functions, broadcasting, …) and also Variables/Constants are defined by a node in a directed graph. The directed edges of the Graph connect nodes to each other and define in which direction information/data flows from one node to the next. There are Input-Nodes where information is fed into the Computation Graph from outside, and Output-Nodes that output the processed data.

After the Graph has been defined, it can be executed on data that is fed into the Graph. Thus, the data *flows* through the graph, changes its content and shape, and is transformed into the output of the Graph. The data can usually be expressed as a multidimensional array, or Tensor, thus the name TensorFlow.

Using this model, it is easy to define the architecture of a neural network using these nodes. Each layer of a neural network can be understood as a special node in the computation graph. There are many pre-defined operations in the TensorFlow API, but users can of course define their own custom operations. But keep in mind that arbitrary computations can be defined using a computation graph, not only operations in the context of machine learning.

Graphs are invoked by as TensorFlow session: `tf.Session()`. A session can take run options as arguments, such as the number of GPUs the graph should be executed on, the specifics of memory allocation on the GPU and what not. Once the necessary data is available, it can be fed into the the computation graph using the `tf.Session.run()` method in which all the magic happens.

### Gradients

In order to train a neural network, using an optimization algorithm such as Stochastic Gradient Descent, we need the definitions of the gradients of all operations in the network. Otherwise, performing backpropagation on the network is not possible. Luckily, TensorFlow offers automatic differentiation for us, such that we only have to define the forward-pass of information through the network. The backward-pass of the error through all layers is inferred automatically. This feature is not unique with TensorFlow — all current ML libraries offer automatic differentiation.



### CUDA

From the start, the focus of TensorFlow was to let the Computing Graph execute on GPUs. Their highly parallel architecture offers ideal performance for excessive matrix-vector arithmetic which is necessary for training machine learning libraries. The NVIDIA CUDA (**C**ompute **U**nified **D**evice **A**rchitecture) API allows TensorFlow to execute **arbitrary** operations on a NVIDIA GPU.

There are also projects with the goal to expose TensorFlow to any OpenCL-compatible device (i.e. also AMD GPUs). However, NVIDIA still remains the clear champion in Deep Learning GPU hardware, not the least due to the success of CUDA+TensorFlow.

Getting a working installation of CUDA on your machine, including CuDNN and the correct NVIDIA drivers for your GPU can be a _painful_ experience (especially since not all TensorFlow versions are compatible with all CUDA/CuDNN/NVIDIA driver versions and you were too lazy to have a look at the version compatibility pages), however, once TensorFlow can use your GPU(s), you will recognize a significant boost in performance.


### Multi GPU support

Large-scale machine learning tasks require access to more than one GPU in order to yield results quickly. Large enough deep neural networks have too many parameters to fit them all into a single GPU. TensorFlow lets users easily declare on which devices (GPU or CPU) the computation graph should be executed.




| ![](/images/tf2.0/2.png) | 
|:--:| 
| *Multi-GPU computation model (source: [https://www.tensorflow.org/tutorials/images/deep_cnn](https://www.tensorflow.org/tutorials/images/deep_cnn))* |




### Eager execution

The TensorFlow Computation Graph is a powerful model for processing information. However, a major point of criticism from the start was the difficulty of debugging such graphs. With statements such as

```python
a = tf.Constant(1.0, dtype=tf.float32)
b = tf.Constant(3.0, dtype=tf.float32)
c = a + b
```


the content of the variable c is not 4.0, as one might expect, but rather a TensorFlow node with no definite value assigned to it yet. The validity of such a statement (and the possible bugs introduced by the statement) can only be tested after the Graph was invoked and a session was run on the Graph.

Thus, TensorFlow released the eager execution mode, for which each node is immediately executed after definition. Statements using tf.placeholder are thus no longer valid. The eager execution mode is simply invoked using `tf.eager_execution()` after importing TensorFlow.

TensorFlow’s eager execution is an imperative programming environment that evaluates operations immediately, without building graphs: operations return concrete values instead of constructing a computational graph to run later. The advantages of this approach are easier debugging of all computations, natural control flow using Python statements instead of graph control flow, and an intuitive interface. The downside of eager mode is the reduced performance since graph-level optimizations such as common subexpression elimination and constant-folding are no longer available.



### Debugger

The TensorFlow Debugger (tfdbg) lets you view the internal structure and states of running TensorFlow graphs during training and inference, which is difficult to debug with general-purpose debuggers such as Python’s dbg to TensorFlow's computation-graph paradigm. It was conceived as an answer to criticism regarding the difficulty in debugging TensorFlow programs. There is both a command-line interface and a Debugging plugin for TensorBoard (more info below) that allows you to inspect the computation graph for debugging. For a detailed introduction, please find [https://www.tensorflow.org/guide/debugger](https://www.tensorflow.org/guide/debugger).

### TensorBoard

You can use TensorBoard to visualize your TensorFlow graph, plot quantitative metrics about the execution of your graph, and show additional data such as images that pass through it during training or inference. It is definitely the way to go if you wish to visualize any kind of data that is available during within the computation graph. While TensorBoard was originally introduced as part of TensorFlow, it now lives in its own GitHub repository. However, it will be installed automatically when installing TensorFlow itself.

TensoBoard is not only useful for visualizing training or evaluation data such as losses/accuracies as a function of the number of steps, but also for visualizing image data or sound waveforms. The best way to get an overview of TensorBoard is to have a look at [https://www.tensorflow.org/guide/summaries_and_tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard).

### TPU support

TPUs (Tensor Processing Units) are highly-parallel computing units specifically designed to efficiently process multi-dimensional arrays (a.k.a. **Tensors**), which is particularly useful in Machine Learning. Due to their application-specific integrated circuit (ASIC) design, they are the fastest processors for machine learning applications available today. As of today, Google’s TPUs are proprietary and are not commercially available for any private consumers or businesses. They are part of the Google Compute Engine, where you can rent compute instances that have access to TPUs for your large-scale machine learning needs. Needless to say that Google aims at making every TensorFlow operation executable on a TPU device to further strengthen its position in the ever-growing cloud computing market.

You can, however, test the performance of a single TPU for yourself in Google Colab, a platform that can host and execute Jupyter Notebooks, with access to CPU/GPU or TPU instances on the Google Compute Engine, for free! For a small introduction, click [here](https://medium.com/@jannik.zuern/using-a-tpu-in-google-colab-54257328d7da).


### TensorRT


While neural network training typically happens on powerful hardware with sometimes multiple GPUs, neural network inference usually happens locally on consumer devices (unless the raw data is streamed to another cloud service, and inference happens there) such as the onboard computers of autonomous cars or even mobile phones. NVIDIA offers a module called TensorRT that takes a TensorFlow Graph of a trained neural network expressed using the TensorFlow API and converts it to a Computation Graph specifically optimized for inference. This usually results in a significant performance gain compared to inference within TensorFlow itself. For an introduction to TensorRT, click [here](https://medium.com/@jannik.zuern/exploring-the-new-tensorflow-integration-of-nvidias-tensorrt-148ee0d95cd5).

### tf.contrib

TensorFlow has a vibrant community on GitHub that added quite some functionality to the core and the peripherals of TensorFlow (obviously a strong argument for Google to open-source TensorFlow). Most of these modules are collected in the `tf.contrib` module. Due to the high market share of TensorFlow, quite a few modules can be found here that you would otherwise have to implement yourself.

### Tensorflow Hub

TensorFlow Hub is a library for the publication, discovery, and consumption of reusable parts of machine learning models. A module is a **self-contained piece** of a TensorFlow graph, along with its weights and assets, that can be reused across different tasks in a process known as transfer learning. Fore more details please find [https://www.tensorflow.org/hub](https://www.tensorflow.org/hub).

### More, more, more

There is so much more to talk about. Which components of the TensorFlow ecosystem should at least be mentioned?

- **TensorFlow Docker container**: Docker containers containing pre-installed TensorFlow, including CUDA compatibility for graph execution on GPUs from within the Docker container
- **TensorFlow Lite**: TensorFlow Lite is an open source deep learning framework for on-device inference on devices such as embedded systems and mobile phones.
- **TensorFlow Extended (TFX)**: TFX is a Google-production-scale machine learning platform based on TensorFlow. It provides a configuration framework and shared libraries to integrate common components needed to define, launch, and monitor your machine learning system.



### What _sucks_?

One of the strengths of TensorFlow, the Computation Graph, is arguably also one of its weaknesses. While the static computation Graph definitely boosts performance (since graph-level optimizations can happen after the graph is built and before it is executed), it also makes debugging the graph difficult and cumbersome — even with tools such as the TensorFlow Debugger. Also, benchmarks have shown that several other frameworks can compete on equal terms with TensorFlow, while keeping a simpler syntax. Additionally, first building a graph and then instantiating it using tf.Sessions is not very intuitive and definitely scares or bewilders some inexperienced users.

The TensorFlow API arguably also has weaknesses, i.e. discussed here. Some users complain about the low-level-feeling when using the TensorFlow API, even when solving a high-level task. Much boilder-plate code is needed for simple tasks such as training a linear classifier.

# TensorFlow 2.0 — What’s new?

After delving into the depths of TensorFlow 1.XX, what will change with the big 2? Did the TensorFlow team respond to some of the criticism of the past? And what justifies calling it version 2.0, and not 1.14?

In multiple blog posts and announcements, some of the future features of TF2.0 have been revealed. Also, the TF2.0 API reference lists have already been made publicly available. While TF2.0 is still in alpha version, it is expected that the official Beta, Release Candidates, and the final release will be made available later this year.

Let’s have a closer look at some of the **novelties** of TF2.0:

### Goodbye `tf`, hello `tf.keras`

For a while, TensorFlow has offered the tf.keras API as part of the TensorFlow module, offering the same syntax as the Keras machine learning library. Keras has received much praise for its simple and intuitive API for defining network architectures and training them. Keras integrates tightly with the rest of TensorFlow so you can access TensorFlow’s features whenever you want. The Keras API makes it easy to get started with TensorFlow. Importantly, Keras provides several model-building APIs (Sequential, Functional, and Subclassing), so you can choose the right level of abstraction for your project. TensorFlow’s implementation contains enhancements including eager execution, for immediate iteration and intuitive debugging, and `tf.data`, for building scalable input pipelines.

### tf.data

Training data is read using input pipelines which are created using tf.data. This will be the preferred way of declaring input pipelines. Pipelines using tf.placeholders and feed dicts for sessions will still work under the TensorFlow v1 compatibility mode, but will no longer benefit from performance improvements in subsequent tf2.0 versions.

### Eager execution default

TensorFlow 2.0 runs with eager execution (discussed previously) by default for ease of use and smooth debugging.


### RIP `tf.contrib`

Most of the modules in tf.contrib will depreciate in tf2.0 and will be either moved into core TensorFlow or removed altogether.


### `tf.function` decorator

The tf.function function decorator transparently translates your Python programs into TensorFlow graphs. This process retains all the advantages of 1.x TensorFlow graph-based execution: Performance optimizations, remote execution and the ability to serialize, export and deploy easily while adding the flexibility and ease of use of expressing programs in simple Python. In my opinion, this is the biggest change and paradigm shift from v1.X to v2.0.


### No more `tf.Session()`

When code is eagerly executed, sessions instantiating and running computation graphs will no longer be necessary. This simplifies many API calls and removes some boilerplate code from the codebase.

### TensorFlow 1.XX legacy

It will still be possible to run tf1.XX code in tf2 without any modifications, but this does not let you take advantage of many of the improvements made in TensorFlow 2.0. Instead, you can try running a conversion script that automatically converts the old tf1.XX calls to tf2 calls, if possible. The detailed migration guide from tf1 to tf2 will give you more information if needed.


--------------------


I hope you liked this small overview, and see you next time!

Happy Tensorflowing!



# Further reading

- [Effective TF2](https://www.tensorflow.org/alpha/guide/effective_tf2)
- [Whats new in TF2?](https://medium.com/tensorflow/whats-coming-in-tensorflow-2-0-d3663832e9b)
