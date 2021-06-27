# The annotated MNIST image classification example with Flax Linen and Optax

_Author: @8bitmp3_

This tutorial uses [Flax](https://flax.readthedocs.io)—a high-performance deep learning library for [JAX](https://jax.readthedocs.io) designed for flexibility—to show you how to construct a simple convolutional neural network (CNN) using the Linen API and [Optax](https://github.com/deepmind/optax/) and train the network for image classification on the MNIST dataset.

If you're new to JAX, check out:

- [JAX quickstart](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
- [Thinking in JAX](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html)
- [JAX 101](https://jax.readthedocs.io/en/latest/jax-101/index.html)
- [JAX for the impatient](https://flax.readthedocs.io/en/latest/notebooks/jax_for_the_impatient.html)

To learn more about Flax and its Linen API, refer to:
- [Flax basics](https://flax.readthedocs.io/en/latest/notebooks/flax_basics.html)
- [Flax patterns: Managing state and parameters](https://flax.readthedocs.io/en/latest/patterns/state_params.html)
- [Linen design principles](https://flax.readthedocs.io/en/latest/design_notes/linen_design_principles.html)
- [Linen introduction](https://github.com/google/flax/blob/master/docs/notebooks/linen_intro.ipynb)
- [More notebooks](https://github.com/google/flax/tree/master/docs/notebooks) (including a more concise version of this MNIST notebook by @andsteing)

This tutorial has the following workflow:

- Perform a quick setup
- Build a convolutional neural network model with the Linen API that classifies images
- Define a loss and accuracy metrics function
- Create a dataset function with TensorFlow Datasets
- Define training and evaluation functions
- Load the MNIST dataset
- Initialize the parameters with PRNGs and instantiate the optimizer with Optax
- Train the network and evaluate it

If you're using [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb) (Colab), enable the GPU acceleration (**Runtime** > **Change runtime type** > **Hardware accelerator**:**GPU**).

## Setup

1. Import JAX, [JAX NumPy](https://jax.readthedocs.io/en/latest/jax.numpy.html), Flax, [Optax](https://github.com/deepmind/optax/), ordinary NumPy, and TensorFlow Datasets (TFDS). Flax can use any data-loading pipeline and this example demonstrates how to utilize TFDS.

```python
!pip install --upgrade -q pip jax jaxlib flax optax tensorflow-datasets
```

2. Import JAX, [JAX NumPy](https://jax.readthedocs.io/en/latest/jax.numpy.html) (which lets you run code on GPUs and TPUs), Flax, ordinary NumPy, and TFDS. Flax can use any data-loading pipeline and this example demonstrates how to utilize TFDS.

```python
import jax
import jax.numpy as jnp               # JAX NumPy

from flax import linen as nn          # The Linen API
from flax.training import train_state
import optax                          # The Optax gradient processing and optimization library

import numpy as np                    # Ordinary NumPy
import tensorflow_datasets as tfds    # TFDS for MNIST
```

## Build a model

Build a convolutional neural network with the Flax Linen API by subclassing [`flax.linen.Module`](https://flax.readthedocs.io/en/latest/flax.linen.html#core-module-abstraction). Because the architecture in this example is relatively simple—you're just stacking layers—you can define the inlined submodules directly within the `__call__` method and wrap it with the `@compact` decorator ([`flax.linen.compact`](https://flax.readthedocs.io/en/latest/flax.linen.html#compact-methods)).

```python
class CNN(nn.Module):

  @nn.compact
  # Provide a constructor to register a new parameter 
  # and return its initial value
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1)) # Flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)    # There are 10 classes in MNIST
    return x
```

## Create a metrics function

For loss and accuracy metrics, create a separate function:

  - Optax has a built-in softmax cross-entropy loss ([`optax.softmax_cross_entropy`](https://optax.readthedocs.io/en/latest/api.html#optax.softmax_cross_entropy)). You will be defining and computing the loss inside a training step function later as follows:
  - The labels can be one-hot encoded with [`jax.nn.one_hot`](https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.one_hot.html), as demonstrated below.

```python
def compute_metrics(logits, labels):
  loss = jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, num_classes=10)))
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy
  }
  return metrics
```

## The dataset

Define a function that:
  - Uses TFDS to load and prepare the MNIST dataset; and
  - Converts the samples to floating-point numbers.

```python
def get_datasets():
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  # Split into training/test sets
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  # Convert to floating-points
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.0
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.0
  return train_ds, test_ds
```

## Training and evaluation functions

1. Write a training step function that:

  - Evaluates the neural network given the parameters and a batch of input images with the [`flax.linen.Module.apply`](https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.apply) method.
  - Defines and computes the `cross_entropy_loss` function.
  - Evaluates the loss function and its gradient using [`jax.value_and_grad`](https://jax.readthedocs.io/en/latest/jax.html#jax.value_and_grad) (check the [JAX autodiff cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#Evaluate-a-function-and-its-gradient-using-value_and_grad) to learn more).
  - Applies a [pytree](https://jax.readthedocs.io/en/latest/pytrees.html#pytrees-and-jax-functions) of gradients (`flax.training.train_state.TrainState.apply_gradients`) to the optimizer to update the model's parameters.
  - Returns the optimizer `state` and computes the metrics using `compute_metrics` (defined earlier).

  Use JAX's [`@jit`](https://jax.readthedocs.io/en/latest/jax.html#jax.jit) decorator to trace the entire `train_step` function and just-in-time(JIT-compile with [XLA](https://www.tensorflow.org/xla) into fused device operations that run faster and more efficiently on hardware accelerators.

```python
@jax.jit
def train_step(state, batch):
  def loss_fn(params):
    logits = CNN().apply({'params': params}, batch['image'])
    loss = jnp.mean(optax.softmax_cross_entropy(
        logits=logits, 
        labels=jax.nn.one_hot(batch['label'], num_classes=10)))
    return loss, logits
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits, batch['label'])
  return state, metrics
```

2. Create a [`jit`](https://jax.readthedocs.io/en/latest/jax.html#jax.jit)-compiled function that evaluates the model on the test set using [`flax.linen.Module.apply`](https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.apply):

```python
@jax.jit
def eval_step(params, batch):
  logits = CNN().apply({'params': params}, batch['image'])
  return compute_metrics(logits, batch['label'])
```

3. Define a training function for one epoch that:

  - Shuffles the training data before each epoch using [`jax.random.permutation`](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.permutation.html) that takes a PRNGKey as a parameter (discussed in more detail later in this tutorial and in [JAX - the sharp bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#JAX-PRNG)).
  - Runs an optimization step for each batch.
  - Retrieves the training metrics from the device with `jax.device_get` and computes their mean across each batch in an epoch.
  - Returns the optimizer `state` with updated parameters and the training loss and accuracy metrics (`training_epoch_metrics`).

```python
def train_epoch(state, train_ds, batch_size, epoch, rng):
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, len(train_ds['image']))
  perms = perms[:steps_per_epoch * batch_size]  # Skip an incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  batch_metrics = []

  for perm in perms:
    batch = {k: v[perm, ...] for k, v in train_ds.items()}
    state, metrics = train_step(state, batch)
    batch_metrics.append(metrics)

  training_batch_metrics = jax.device_get(batch_metrics)
  training_epoch_metrics = {
      k: np.mean([metrics[k] for metrics in training_batch_metrics])
      for k in training_batch_metrics[0]}

  print('Training - epoch: %d, loss: %.4f, accuracy: %.2f' % (epoch, training_epoch_metrics['loss'], training_epoch_metrics['accuracy'] * 100))

  return state, training_epoch_metrics
```

4. Create a model evaluation function that:

  - Evalues the model on the test set.
  - Retrieves the evaluation metrics from the device with `jax.device_get`.
  - Copies the metrics [data stored](https://flax.readthedocs.io/en/latest/design_notes/linen_design_principles.html#how-are-parameters-represented-and-how-do-we-handle-general-differentiable-algorithms-that-update-stateful-variables) in a JAX [pytree](https://jax.readthedocs.io/en/latest/pytrees.html#pytrees-and-jax-functions).
  - Returns the test loss and accuracy.

```python
def eval_model(model, test_ds):
  metrics = eval_step(model, test_ds)
  metrics = jax.device_get(metrics)
  eval_summary = jax.tree_map(lambda x: x.item(), metrics)
  return eval_summary['loss'], eval_summary['accuracy']
```

## Load the dataset

Download the dataset and preprocess it with `get_datasets` you defined earlier:

```python
train_ds, test_ds = get_datasets()
```

## Initialize the parameters with PRNGs and instantiate the optimizer

1. [PRNGs](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#JAX-PRNG): Before you start training the model, you need to randomly initialize the parameters.

  In NumPy, you would usually use the stateful pseudorandom number generators (PRNG). 
  
  JAX, however, uses an explicit PRNG (refer to [JAX - the sharp bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#JAX-PRNG) for details):

  - Get one [PRNGKey](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.PRNGKey.html#jax.random.PRNGKey).
  - [`split`](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.split.html#jax.random.split) it to get a second key that you'll use for parameter initialization.

  Note that in JAX and Flax you can have [separate PRNG chains](https://flax.readthedocs.io/en/latest/design_notes/linen_design_principles.html#how-are-parameters-represented-and-how-do-we-handle-general-differentiable-algorithms-that-update-stateful-variables) (with different names, such as `rng` and `init_rng` below) inside `Module`s for different applications. (Learn more about [PRNG chains](https://flax.readthedocs.io/en/latest/design_notes/linen_design_principles.html#how-are-parameters-represented-and-how-do-we-handle-general-differentiable-algorithms-that-update-stateful-variables) and [JAX PRNG design](https://github.com/google/jax/blob/master/design_notes/prng.md).)

```python
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
```

2. Instantiate the `CNN` model and initialize its parameters using a PRNG:

```python
cnn = CNN()
params = cnn.init(init_rng, jnp.ones([1, 28, 28, 1]))['params']
```

3. Instantiate the [SGD optimizer with Optax](https://optax.readthedocs.io/en/latest/api.html#sgd):

```python
nesterov_momentum = 0.9
tx = optax.sgd(learning_rate=learning_rate, nesterov=nesterov_momentum)
```

4. Create a [`TrainState`](https://flax.readthedocs.io/en/latest/flip/1009-optimizer-api.html#train-state) data class that applies the gradients and updates the optimizer state and parameters.

```python
state = train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)
```

## Train the network and evaluate it

1. Set the default number of epochs and the size of each batch:

```python
num_epochs = 10
batch_size = 32
```

2. Finally, begin training and evaluating the model over 10 epochs:

  - For your training function (`train_epoch`), you need to pass a PRNG key used to permute image data during shuffling. Since you have created a PRNG key when initializing the parameters in your nework, you just need to [split](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#JAX-PRNG) or "fork" the PRNG state into two (while maintaining the usual desirable PRNG properties) to get a new subkey (`input_rng`, in this example) and the previous key (`rng`). Use [`jax.random.split`](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.split.html#jax.random.split) to carry this out. (Learn more about [JAX PRNG design](https://github.com/google/jax/blob/master/design_notes/prng.md).)
  - Run an optimization step over a training batch (`train_epoch`).
  - Evaluate on the test set after each training epoch (`eval_model`).
  - Retrieve the metrics from the device and print them.

```python
for epoch in range(1, num_epochs + 1):
  # Use a separate PRNG key to permute image data during shuffling
  rng, input_rng = jax.random.split(rng)
  # Run an optimization step over a training batch
  state, train_metrics = train_epoch(state, train_ds, batch_size, epoch, input_rng)
  # Evaluate on the test set after each training epoch
  test_loss, test_accuracy = eval_model(state.params, test_ds)
  print('Testing - epoch: %d, loss: %.2f, accuracy: %.2f' % (epoch, test_loss, test_accuracy * 100))
```
    Training - epoch: 1, loss: 0.1963, accuracy: 93.96
    Testing - epoch: 1, loss: 0.09, accuracy: 96.97
    Training - epoch: 2, loss: 0.0622, accuracy: 98.10
    Testing - epoch: 2, loss: 0.05, accuracy: 98.35
    Training - epoch: 3, loss: 0.0428, accuracy: 98.70
    Testing - epoch: 3, loss: 0.04, accuracy: 98.74
    Training - epoch: 4, loss: 0.0330, accuracy: 98.98
    Testing - epoch: 4, loss: 0.03, accuracy: 99.02
    Training - epoch: 5, loss: 0.0263, accuracy: 99.16
    Testing - epoch: 5, loss: 0.03, accuracy: 99.03
    Training - epoch: 6, loss: 0.0219, accuracy: 99.31
    Testing - epoch: 6, loss: 0.03, accuracy: 99.00
    Training - epoch: 7, loss: 0.0178, accuracy: 99.44
    Testing - epoch: 7, loss: 0.03, accuracy: 99.03
    Training - epoch: 8, loss: 0.0139, accuracy: 99.58
    Testing - epoch: 8, loss: 0.03, accuracy: 99.08
    Training - epoch: 9, loss: 0.0116, accuracy: 99.66
    Testing - epoch: 9, loss: 0.03, accuracy: 99.16
    Training - epoch: 10, loss: 0.0102, accuracy: 99.70
    Testing - epoch: 10, loss: 0.03, accuracy: 99.01


Once the training and testing is done after 10 epochs, the output should show that your model was able to achieve approximately 99% accuracy.
