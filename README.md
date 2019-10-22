# Probabilistic Inference for Learning Control (PILCO)
[![Build Status](https://travis-ci.org/nrontsis/PILCO.svg?branch=master)](https://travis-ci.org/nrontsis/PILCO)
[![codecov](https://codecov.io/gh/nrontsis/PILCO/branch/master/graph/badge.svg)](https://codecov.io/gh/nrontsis/PILCO)

A modern \& clean implementation of the [PILCO](https://ieeexplore.ieee.org/abstract/document/6654139/) Algorithm in `TensorFlow`.

Unlike PILCO's [original implementation](http://mlg.eng.cam.ac.uk/pilco/) which was written as a self-contained package of `MATLAB`, this repository aims to provide a clean implementation by heavy use of modern machine learning libraries.

In particular, we use `TensorFlow` to avoid the need for hardcoded gradients and scale to GPU architectures. Moreover, we use [`GPflow`](https://github.com/GPflow/GPflow) for Gaussian Process Regression.

The core functionality is tested against the original `MATLAB` implementation.

## Example of usage
Before using, or installing, PILCO, you need to have `Tensorflow 1.13.1` installed (either the gpu or the cpu version). It is recommended to install everything in a fresh `conda` environment with `python>=3.7`. Given `Tensorflow`, PILCO can be installed as follows
```
git clone https://github.com/nrontsis/PILCO && cd PILCO
python setup.py develop
```

The examples included in this repo use [`OpenAI gym 0.15.3`](https://github.com/openai/gym#installation) and [`mujoco-py 2.0.2.7`](https://github.com/openai/mujoco-py#install-mujoco). Once these dependencies are installed, you can run one of the examples as follows
```
python examples/inverted_pendulum.py
```
While running an example, `Tensorflow` might print a lot of warnings, [some of which are deprecated](https://github.com/tensorflow/tensorflow/issues/25996). If necessary, you can suppress them by running
```python
tf.logging.set_verbosity(tf.logging.ERROR)
```
right after including `TensorFlow` in Python.

## Credits:

The following people have been involved in the development of this package:
* [Nikitas Rontsis](https://github.com/nrontsis)
* [Kyriakos Polymenakos](https://github.com/kyr-pol)

## References

See the following publications for a description of the algorithm: [1](https://ieeexplore.ieee.org/abstract/document/6654139/), [2](http://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf),
[3](https://pdfs.semanticscholar.org/c9f2/1b84149991f4d547b3f0f625f710750ad8d9.pdf)

