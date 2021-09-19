# Empirical study of the modulus as activation function in computer vision applications

This repository contains the code of the study "Empirical study of the modulus as activation function in computer vision applications". Please follow the instructions below to reproduce the results.

This readme file will be completed once the study has been published.

## Getting started
If you are interested in run the code, please, follow the next steps.

1. Install [pyenv](https://github.com/pyenv/pyenv) and [poetry](https://python-poetry.org/) in your system following the linked official guides.
2. Open a terminal, clone this repository and `cd` to the cloned folder.
3. Run `pyenv install $(cat .python-version)` in your terminal for installing the required python version
4. Configure poetry with `poetry config virtualenvs.in-project true`
5. Create the virtual environment with `poetry install`
6. Activate the environment with `source .venv/bin/activate`

The `batcher.sh` script contains the code to run all the experiments for a given random seed. Run `bash batcher.sh <seed>` replacing `<seed>` by an integer to run the experiments. The seeds that have been used for generating the current results are the following ones `655321`, `655322`, ..., `655350`.

## Additional experiments
Apart from the experiments described in this study, we have tested several things informally and others did not produce good results. The findings reported in this subsection may not be conclusive and may need more testing to get more robust conclusions. Below, we try to briefly summarize this part of the work.
- The benchmark nonlinearities used to report the results are a subset of our initial pool of activations. We informally tested SELU, GELU, Sigmoid and SoftPlus activations concluding that they did not work better than our proposal, although we did not run repeated experiments.
- Other smooth versions of the modulus nonlinearity were tested, such as $y=\log[\cosh(\beta x)]$ where $\beta$ is a strictly positive hyperparameter that controls the slope of the modulus approximation, concluding that the training process was not stable. In some cases the gradients vanished and in others they exploded. 
- Asymmetric versions of the modulus function were tested (for example $y=max(\delta x, -\beta x)$ where $\delta, \beta$ are strictly positive hyperparameters). We did not see any benefit out of these trials.
- We tried sinusoidal activation functions and, although the training process seemed to be stable, the performance was lower than the benchmark activation functions.

## Contribution
Pull requests and issues will be tackled upon availability.

## License
This repository is licensed under MIT license. More info in the LICENSE file. Copyright (c) 2021 Iván Vallés Pérez
