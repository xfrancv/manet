# MANET: MArkov NETwork learning library

This package implements maximum-margin based algorithms for learning Markov Network classifiers from examples. The package supports linearly parameterized MN classifiers with arbitrarily complex pair-wise interactions between labels. The inference of the labels leads to solving

$$\hat{\mathbf{y}} \in {\rm Arg}\max_{\mathbf{y}\in\cal{Y}^{\cal{V}}} \left(\sum_{v\in\cal{V}} \mathbf{w}_{y_v}^T \mathbf{x}_v + \sum_{v,v'\in\cal{E}}g(y_v,y_{v'})\right )$$

where $\cal{V}$ is a set of objects, $\cal{E}\subset\left(\cal{V}\atop 2\right)$ set of edges, $(\mathbf{x}_v\in\Re^n\mid v\in\cal{V})$ sequence of observable features, $\hat{\mathbf{y}}\in\cal{Y}^{\cal{V}}$ sequence of predicted labels. The undirected graph of interactions $(\cal{V},\cal{E})$ is provided by user and it can have an arbitrary structure. The weights $\mathbf{w}_y, y\in\cal{Y}$, and the label score $g\colon\cal{Y}\times\cal{Y}\rightarrow\Re$ are parameters learned from examples. 

MANET implements M3N algorithm [1][2] for learning MN classifiers from examples $(\mathbf{x}^1,\mathbf{y}^1),\ldots,(\mathbf{x}^m,\mathbf{y}^m)$. MANET support learning i) from completely annotated examples when $(\mathbf{x}^i,\mathbf{y}^i) \in\Re^{n\times |\cal{V}|}\times \cal{Y}^{|\cal{V}|}$ and ii) from examples with missing labels when $(\mathbf{x}^i,\mathbf{y}^i) \in\Re^{n\times |\cal{V}|}\times \cal{A}^{|\cal{V}|}$ and $\cal{A}=\cal{Y}\cup ?$ is a label set $\cal{Y}$ extended by a symbol for missing label $?$. 

MANET converts learning of MN classifier into a convex problem via using either Linear Programming Margin-Recaling (LP-MR) loss or Markov Network Adversarial (MANA) loss. The convex problem is tractable by standard SGD or ADAM that are both implemented. 

MANET comes with inference algorithm for case that $(\cal{V},\cal{E})$ is a chain and with Schlesinger's Augmented DAG max-sum solver [3] for case that $(\cal{V},\cal{E})$ is arbitrary. C-implementation of ADAG solver is adopted from [4].


## Requirements

* Python 3.x
* Linux; Tested on Ubuntu 18.04 LTS.

## Install 

Install required python packages:
```bash
pip install numpy pyyaml pandas argparse tqdm cffi scipy scikit-learn matplotlib
```
Compile CFFI interface for C-implementation of the ADAG max-sum solver:
```bash
cd manet/adag_solver/
./build_adag_module.sh
```
Modify your .profile or .bashrc by adding path to the compiled ADAG solver and to the MANET root directory:

```bash
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:MANET_ROOT_DIR/manet/adag_solver/"
PYTHONPATH="${PYTHONPATH}:MANET_ROOT_DIR/"
```

## Getting started

- [Learning to predict sequences](examples/train_hmc.ipynb). This is an example on learning MN classifier predicting label sequences from synthetic examples generated from HMC. It shows how to learn from both completely annotated examples and examples with missing labels. It illustrates all basic functions of the library.

- [Evaluation of M3N algorithm using different proxy losses](examples/ecml2022/README.md). This is an implementation of experiments published in paper [2]. The goal is to evaluate performance of MN classifier learned by M3N algorithm with two different proxies: LP Margin-rescaling loss and MArkov Network Adversarial loss. The proxy losses are evaluated on synthetically generated sequences and on the problem of learning symbolic and visual Sudoku solver.

- [Sudoku solver](examples/sudoku_solver.ipynb). This is an example on using the generic ADAC inference algorithm to implement Sudoku solver.


## Reference
- [1] V.Franc, A.Yermakov. Learning Maximum Margin Markov Networks from examples with missing labels. ACML 2021. 
- [2] V.Franc, D.Prusa, A.Yermakov. Consistent and Tractable Algorithm for Markov Network Learning. ECML PKDD 2022.
- [3] T.Werner. A Linear Programming Approach to Max-sum Problem. A Review. PAMI 2007.
- [4] T.Werner. [LP Relaxation Approach to MAP Inference in Markov Random Fields](https://cmp.felk.cvut.cz/~werner/software/maxsum/)


