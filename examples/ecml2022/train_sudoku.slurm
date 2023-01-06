#!/bin/bash
#SBATCH --job-name=my_array_job   # Job name
#SBATCH --mail-type=ALL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=xfrancv@cmp.felk.cvut.cz   # Where to send mail	
#SBATCH --mem=20gb                   # Job Memory
#SBATCH --output=./logs/array_%A-%a.log    # Standard output and error log
#SBATCH --error=./logs/error_%A-%a.log    # Standard output and error log
#SBATCH --array=0-14    #  for ocr 280                 # Array range
#SBATCH --partition=cpuextralong

pwd; hostname; date
ml tqdm/4.61.2-GCCcore-10.3.0
ml Anaconda3

python3 train.py adam_advhomo sudoku_1000_100_100_mis20 $SLURM_ARRAY_TASK_ID
python3 train.py adam_mrhomo sudoku_1000_100_100_mis20 $SLURM_ARRAY_TASK_ID

python3 train.py adam_mrhomo sudoku_1000_100_100_mis10 $SLURM_ARRAY_TASK_ID
python3 train.py adam_advhomo sudoku_1000_100_100_mis10 $SLURM_ARRAY_TASK_ID

python3 train.py adam_mrhomo sudoku_1000_100_100_mis0 $SLURM_ARRAY_TASK_ID
python3 train.py adam_advhomo sudoku_1000_100_100_mis0 $SLURM_ARRAY_TASK_ID


date
