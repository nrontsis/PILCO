# Probabilistic Inference for Learning Control
[![Build Status](https://travis-ci.org/nrontsis/PILCO.svg?branch=master)](https://travis-ci.org/nrontsis/PILCO)
[![codecov](https://codecov.io/gh/nrontsis/PILCO/branch/master/graph/badge.svg)](https://codecov.io/gh/nrontsis/PILCO)

A modern \& clean implementation of the [PILCO](https://ieeexplore.ieee.org/abstract/document/6654139/) Algorithm in `TensorFlow`.

Unlike PILCO's [original implementation](http://mlg.eng.cam.ac.uk/pilco/) which was written as a self-contained package of `MATLAB`, this repository aims to provide a clean implementation by heavy use of modern machine learning libraries.

In particular, we use `TensorFlow` to avoid the need for hardcoded gradients and scale to GPU architectures. Moreover, we use [`GPflow`](https://github.com/GPflow/GPflow) for Gaussian Process Regression.

The core functionality is tested against the original `MATLAB` implementation.

## Example of usage
First install the package by running:
```
python setup.py develop
```

Then you can run the example of using PILCO in [`OpenAI gym`](https://gym.openai.com) by running:
```
python examples/mountain_car.py
```
or (requires [`Mujoco`](https://github.com/openai/mujoco-py)):
```
python examples/inverted_pendulum.py
```


## Credits:

The following people have been involved in the development of this package:
* [Nikitas Rontsis](https://github.com/nrontsis)
* [Kyriakos Polymenakos](https://github.com/kyr-pol)

## References

See the following publications for a description of the algorithm: [1](https://ieeexplore.ieee.org/abstract/document/6654139/), [2](http://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf),
[3](https://pdfs.semanticscholar.org/c9f2/1b84149991f4d547b3f0f625f710750ad8d9.pdf)
