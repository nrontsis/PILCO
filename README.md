# Probabilistic Inference for Learning Control (PILCO)
[![Build Status](https://travis-ci.org/nrontsis/PILCO.svg?branch=master)](https://travis-ci.org/nrontsis/PILCO)
[![codecov](https://codecov.io/gh/nrontsis/PILCO/branch/master/graph/badge.svg)](https://codecov.io/gh/nrontsis/PILCO)

A modern \& clean implementation of the [PILCO](https://ieeexplore.ieee.org/abstract/document/6654139/) Algorithm in `TensorFlow v2`.

Unlike PILCO's [original implementation](http://mlg.eng.cam.ac.uk/pilco/) which was written as a self-contained package of `MATLAB`, this repository aims to provide a clean implementation by heavy use of modern machine learning libraries.

In particular, we use `TensorFlow v2` to avoid the need for hardcoded gradients and scale to GPU architectures. Moreover, we use [`GPflow v2`](https://github.com/GPflow/GPflow) for Gaussian Process Regression.

The core functionality is tested against the original `MATLAB` implementation.

## Example of usage
Before using `PILCO` you have to install it by running:
```
git clone https://github.com/nrontsis/PILCO && cd PILCO
python setup.py develop
```
It is recommended to install everything in a fresh conda environment with `python>=3.7`

The examples included in this repo use [`OpenAI gym 0.15.3`](https://github.com/openai/gym#installation) and [`mujoco-py 2.0.2.7`](https://github.com/openai/mujoco-py#install-mujoco). Theses dependecies should be installed manually. Then, you can run one of the examples as follows
```
python examples/inverted_pendulum.py
```

## Example Extension: Safe PILCO
As an example of the extensibility of the framework, we include in the folder `safe_pilco_extension` an extension of the standard PILCO algorithm that takes safety constraints (defined on the environment's state space) into account as in [https://arxiv.org/abs/1712.05556](https://arxiv.org/pdf/1712.05556.pdf). The `safe_swimmer_run.py` and `safe_cars_run.py` in the `examples` folder demonstrate the use of this extension.

## Credits:

The following people have been involved in the development of this package:
* [Nikitas Rontsis](https://github.com/nrontsis)
* [Kyriakos Polymenakos](https://github.com/kyr-pol/)

## References

See the following publications for a description of the algorithm: [1](https://ieeexplore.ieee.org/abstract/document/6654139/), [2](http://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf), 
[3](https://pdfs.semanticscholar.org/c9f2/1b84149991f4d547b3f0f625f710750ad8d9.pdf)