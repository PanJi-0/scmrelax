# scmrelax

Estimate combination weights in synthetic control methods by relaxed balancing approaches, including L2, EL and Entropy-SCM-Relaxation.

## Introduction

This python package implements the SCM-relaxation estimator for synthetic control in

- Liao, Shi and Zheng (2025): "Relaxed balancing for synthetic control"

The current package is sufficient to replicate all empirical results in the paper. 

Documentation is provided [here](https://scmrelax.readthedocs.io/en/latest/).
## Installation

```bash
$ pip install git+https://github.com/PanJi-0/scmrelax.git
```

## Usage

The main function of this package is `scmrelax.fit`. It estimates weights using different relaxation methods including empirical likelihood relaxation, entropy relaxation and L2 relaxation. It also provides weights estimated by standard synthetic control; see Abadie, Alberto, and Javier Gardeazabal. 2003. "The Economic Costs of Conflict: A Case Study of the Basque Country ." American Economic Review 93 (1): 113–132.

Here's a step-by-step example demonstrating the usage of the `scmrelax` package:

```
import numpy as np
import scmrelax

# Pre-treatment data (control units)
X_pre = np.array([[1.2, 3.4, 5.6],
                  [2.3, 4.5, 6.7],
                  [3.1, 5.4, 7.8]])

# Pre-treatment target data (treated unit)
y_pre = np.array([2.5, 3.7, 4.1])

# Post-treatment data (control units)
X = np.array([[1.5, 3.8, 6.1],
              [2.7, 4.9, 7.2]])

# Fit the models and get results
results = scmrelax.fit(X_pre, y_pre, X)

# Print results
for method, res in results.items():
    print(f"{method} weights:", res['weights'])
    print(f"{method} predictions:", res['predictions'])
```       

## License

`scmrelax` is contributed by [Chengwang Liao](https://github.com/cwleo), [Ji Pan](https://github.com/PanJi-0), [Yapeng Zheng](https://github.com/YapengZheng). 

It is licensed under the terms of the MIT license.