# Probabilistic Inference for Learning Control
[![Build Status](https://travis-ci.org/nrontsis/PILCO.svg?branch=master)](https://travis-ci.org/nrontsis/PILCO)
[![codecov](https://codecov.io/gh/nrontsis/PILCO/branch/master/graph/badge.svg)](https://codecov.io/gh/nrontsis/PILCO)

A modern \& clean implementation of the [PILCO](https://ieeexplore.ieee.org/abstract/document/6654139/) Algorithm in `TensorFlow`.

Unlike PILCO's [original implementation](http://mlg.eng.cam.ac.uk/pilco/) which was written as a self-contained package of `MATLAB`, this repository aims to provide a clean implementation by heavy use of modern machine learning libraries.

In particular, we use `TensorFlow` to avoid the need for hardcoded gradients and scale to GPU architectures. Moreover, we use [`GPflow`](https://github.com/GPflow/GPflow) for Gaussian Process Regression.

The core functionality is tested against the original `MATLAB` implementation.

## Update to `Tensorflow` 2.0+
Updating to `Tensorflow` and (`gpflow`) 2.0 simplifies the codebase, making several wrapper functions unecessary, and allowing for easier prototyping with eager execution.

## Safe PILCO extension
We extend the standard PILCO algorithm to take safety constraints (defined on the environment's state space) into account as in [https://arxiv.org/abs/1712.05556](https://arxiv.org/pdf/1712.05556.pdf).

## Example of usage
First install the package by running:
```
python setup.py develop
```
Dependencies for the core functionalities can be installed by running:
```
pip install -r requirements.txt
```
However `mujoco_py`, which provides several of the environments in the examples, is not included and should be installed by following the instructions at [mujoco_py](https://github.com/openai/mujoco-py). 

Then you can run the example of using PILCO in [`OpenAI gym`](https://gym.openai.com) by running
```
python examples/inverted_pendulum.py
```

## Credits:

The following people have been involved in the development of this package:
* [Nikitas Rontsis](https://github.com/nrontsis)
* [Kyriakos Polymenakos](https://github.com/kyr-pol/)

## References

See the following publications for a description of the algorithm: [1](https://ieeexplore.ieee.org/abstract/document/6654139/), [2](http://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf), 
[3](https://pdfs.semanticscholar.org/c9f2/1b84149991f4d547b3f0f625f710750ad8d9.pdf)
