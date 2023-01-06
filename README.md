# manet

This package implements algorithms for training Markov Network classifiers from examples. The package supports linearly parameterized MN classifiers with pair-wise interactions between labels:

$$\hat{\mathbf{y}} \in {\rm Arg}\max_{\mathbf{y}\in\cal{Y}^{\cal{V}}} \left(\sum_{v\in\cal{V}} \mathbf{w}_{y_v}^T \mathbf{x}_v + \sum_{v,v'\in\cal{E}}g(y_v,y_{v'})\right )$$

where $\cal{V}$ is a set of objects, $\cal{E}\subset\left(\cal{V}\atop 2\right)$ set of edges, $(\mathbf{x}_v\in\Re^n\mid v\in\cal{V})$ sequence of observable features, $\hat{\mathbf{y}}\in\cal{Y}^{\cal{V}}$ sequence of predicted labels. The undirected graph of interactions $(\cal{V},\cal{E})$ is fixed apriory and it can be arbitrary structure. 


## Requirements

* Python 3.x

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

