# spa\_capacity\_analysis

Code for analyzing the capacity of distributed representations accompanying our paper [Analyzing the Capacity of Distributed Vector Representations to Encode Spatial Information](https://ieeexplore.ieee.org/document/9207137). You can find the pdf of the paper on [arXiv](https://arxiv.org/pdf/2010.00055.pdf).
We recommend using Python 3.x.

## Dependencies

We rely only on standard python packages except for [nengo-spa](https://www.nengo.ai/nengo-spa/), which is used to generate vocabularies and manipulate vectors.
To install all required packages, run the following command

```pip install numpy matplotlib pandas seaborn nengo_spa```

## Scripts

This repository contains three python scripts.
To simply re-generate the plots from the paper, please run

```python create_plots.py```

This will create the plots from the paper in a sub-folder named "plots".

The other two scripts `spa_superposition_capacity.py` and `spa_power_saturation_analysis.py` can be used to generate new evaluatin data for the superposition and convolutive power analysis respectively (as mentioned in the paper). 
Each of the scripts generates a new pandas frame containing the experimental results, which is saved in the "data" sub-folder.
