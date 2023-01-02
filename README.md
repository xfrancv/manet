# manet

## Requirements

* Python 3.x

## Install 

Install required python packages:
```bash
pip install numpy pyyaml pandas argparse tqdm cffi scipy
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

