# manet

This example shows how to train a linear Markov Network predictor that outputs a sequence of labels
$$
   (\hat{y}_0,\ldots,\hat{y}_{l-1})\in\argmax_{(y_0,\ldots,y_{l-1})\in \cal{Y}^{l}} \left( \sum_{v=0}^{l-1} \mathbf{x}_v^T\mathbf{w}_{y_v} + \sum_{v=1}^{l-1} g(y_{v-1},y_v) \right )
$$
where $\cal{Y}=\{0,1,\ldots,n_y-1\}$ is set of $n_y$ labels, $l$ is length of the sequence, $(\mathbf{x}_0,\ldots,\mathbf{x}_{l-1})\in\R^{n\times l}$ is a sequence of input feature vectors, $(\hat{y}_0,\ldots,\hat{y}_{l-1})\in\cal{Y}^l$ is predicted sequence of labels, $\mathbf{w}_y\in\R^n$, $y\in\cal{Y}$, are weight vectors and $g\colon\cal{Y}\times\cal{Y}\rightarrow\R$ is a function scoring quality of a pair of consecutive labels. The weight vectors $\mathbf{w}_y\in\R^n$, $y\in\cal{Y},$ and the pair-wise score $g\colon\cal{Y}\times\cal{Y}\rightarrow\R$ are the parameters to be learned from examples.

This example shows how to learn the parameters from completly annotated sequences (standard supervised learning):
$$
   \{ ((\mathbf{x}_0^j,\ldots,\mathbf{x}_{l-1}^j), (y_0^j,\ldots,y_{l-1}^j))\in \R^{n\times l}\times \cal{Y}^l  \mid j = 1,\ldots, m \}
$$
However, we also show how learn the parameters from partially annoated examples when some labels in the training set are missing:
$$
   \{ ((\mathbf{x}_0^j,\ldots,\mathbf{x}_{l-1}^j), (y_0^j,\ldots,y_{l-1}^j))\in \R^{n\times l}\times (\cal{Y}\cup \{?\})^l  \mid j = 1,\ldots, m \}
$$
The algorithm implemented in MANET is guaranteed to work under assumption that the labels are missing at random.

The objective is to learn MN predictor with minimal expected Hamming loss. Becasue the Hamming loss is hard to optimize (its gradient is zero almost everywhere), the M3N algorithm replaces the target loss by easier to optimize proxz loss. MANET implements two proxy losses: i) LP relaxed Margin-Rescaling loss and ii) MArkov Network Adversarial (MANA) loss.

In this example, the input/output sequences are generated from a known Hidden Markov Chain model. Hence, we can estimate the Bayes risk and use it as a reference solution when evaluating the MN predictor learned from data.


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

