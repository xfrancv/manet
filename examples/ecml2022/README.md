# ECML2022
This folder replicates experiments from paper: 

Franc, Prusa,yermakov. Consistent and tractable learning of Markov Networks. ECML 2022.

The goal is to evaluate performance of MN classifier learned by M3N algorithm with two different proxies: LP Margin-rescaling loss and MArkov Network Adversarial loss. The proxy losses are evaluated on synthetically generated sequences and on the problem of learning symbolic and visual Sudoku solver.

## 1. Create data
Sequences of observable and hidden labels generated from known Hidden Markov Chain:
```bash
python3 create_hmc_dataset.py
```

Examples of symbolic Sudoku puzzles with solutions:
```bash
python3 create_sudoku_dataset.py
```

Examples of visual Sudoku puzzles created from MNIST digits along with solutions:
```bash
python3 create_visual_sudoku_dataset.py
```

Each dataset contains of 5 splits of the examples into training/validation and test part. The number of splits and the number of trn/val/tst examples can specified in the header of the scripts.

For each dataset, the scripts generate examples with different amount of (randomly) missing labels. The amount of missing labels is set to 0%, 10% and 20%, however it can be modified in the header of the scripts.

## 2. Run training 

The configuration of experiments for M3N algorithm with MANA loss is [config/adam_advhomo.yaml](config/adam_advhomo.yaml). The configuration M3N algoritm with Margin-rescaling loss is [config/adam_mrhomo.yaml](config/adam_mrhomo.yaml).
The regularization constants to try are defined by item <code>lambda</code> and the sizes of training set by <code>n_examples</code>. 


On a computer with single CPU run the following scripts:
```bash
./train_hmc.sh
./train_sudoku.sh
./train_visual_sudolu.sh
```
On computer cluster with SLURM invoke the following scripts:
```bash
sbatch -n15 train_hmc.slurm
sbatch -n15 train_sudoku.slurm
sbatch -n15 train_visula_sudoku.slurm
```

## 3. Evaluation

On a computer with single CPU run the following scripts:
```bash
./eval_hmc.sh
./eval_sudoku.sh
./eval_visual_sudolu.sh
```
On computer cluster with SLURM invoke the following scripts:
```bash
sbatch -n15 eval_hmc.slurm
sbatch -n15 eval_sudoku.slurm
sbatch -n15 eval_visula_sudoku.slurm
```

## 4. Show results

Result visualization is in [show_results.ipynb](show_results.ipynb).