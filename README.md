# Probabilistic Inference for Learning Control
This is a `TensorFlow` implementation of the [PILCO](https://ieeexplore.ieee.org/abstract/document/6654139/) Reinforcement Learning Algorithm. PILCO's [original implementation](http://mlg.eng.cam.ac.uk/pilco/) is in `MATLAB`. This repository aims to provide a modern \& significantly cleaner implementation.

By using `TensorFlow` we are able to greatly simplify the code, avoiding the need for hardcoded gradients. At the same time, we avoid re-implementing Gaussian Process Regression but instead rely on the [`GPflow`](https://github.com/GPflow/GPflow) library.

The core functionality is tested against the MATLAB implementation. Tests can be invoked via `pytest`.

A minimal example of using PILCO in [`OpenAI gym`](https://gym.openai.com) can be found in `examples/inverted_pendulum.py`. You can run it with:

```
python -m PILCO.examples.inverted_pendulum.py
```