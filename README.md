# scmrelax

Estimate combination weights by relaxation approaches to synthetic control, including L2, EL and entropy-SCM-Relaxation.

## Introduction

This python package implements the SCM-relaxation estimator for synthetic control in

- Liao, Shi and Zheng (2025): "A Relaxation Approach to Synthetic Control" ([arxiv](https://arxiv.org/abs/2508.01793))

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
## Replication of empirical results
We replicate the empirical application in Liao, Shi and Zheng (2025) using the current package. In docs file set, you can find the following files 
```
.
├── GDP_application_2016.ipynb     # Main analysis notebook for 2016 Brexit referendum
├── GDP_application_2020.ipynb     # Robustness check analysis notebook
├── balanced_GDP_data.csv          # Balanced GDP dataset
```
### Data Description

The analysis uses real GDP data from the [CEIC](https://www.ceicdata.com/en) database. The dataset `balanced_GDP_data.csv` contains:

- Quaterly GDP data for multiple countries
- Time period: 2002Q4-2024Q2
- Treatment time: 2016Q3
- Treatment unit: United Kingdom
- Donor pool: Other countries in the dataset

### Usage

1. Required Python packages:
   - pandas
   - numpy
   - scipy
   - matplotlib
   - jupyter
   - scmrelax

2. To run the analysis:
   - Open `GDP_application_2016.ipynb` in Jupyter Notebook
   - Execute cells sequentially
  
3. To run the robustness check:
   - Open `GDP_application_2020.ipynb` in Jupyter Notebook
   - Execute cells sequentially

### Results

The analysis examines the impact of the 2016 Brexit referendum on the UK's GDP growth rate using SCM and SCM-relaxation estimators. We also use the fitted growth rates to extrapolate the GDP and estimate the ATEs. Key results include:

- Comparison of estimated weights between SCM and L2-SCM-relaxation
- Average treatment effect of Brexit on the UK's GDP as a percentage of the UK's total GDP after 2016Q3
- Robustness checks when we specify the treatment time as 2020Q1

## Citation

If you find the code and data useful, please consider citing:

```
@misc{liao2025relaxationapproachsyntheticcontrol,
      title={A Relaxation Approach to Synthetic Control}, 
      author={Chengwang Liao and Zhentao Shi and Yapeng Zheng},
      year={2025},
      eprint={2508.01793},
      archivePrefix={arXiv},
      primaryClass={econ.EM},
      url={https://arxiv.org/abs/2508.01793}, 
}
```

## License

`scmrelax` is contributed by [Chengwang Liao](https://github.com/cwleo), [Ji Pan](https://github.com/PanJi-0), [Yapeng Zheng](https://github.com/YapengZheng). 

It is licensed under the terms of the MIT license.
