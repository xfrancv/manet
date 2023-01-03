#!/bin/bash
#SBATCH --job-name=my_array_job   # Job name
#SBATCH --mail-type=ALL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=xfrancv@cmp.felk.cvut.cz   # Where to send mail	
#SBATCH --mem=20gb                   # Job Memory
#SBATCH --output=./logs/array_%A-%a.log    # Standard output and error log
#SBATCH --error=./logs/error_%A-%a.log    # Standard output and error log
#SBATCH --partition=cpuextralong
#SBATCH --array=0-165                 # Array range

pwd; hostname; date
ml tqdm/4.61.2-GCCcore-10.3.0
ml Anaconda3

#export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/home/xfrancv/Work/pymaxsum/manet/adag_solver"

python3 eval.py adam_advhomo hmc_30x30x100_mis0 1000 $SLURM_ARRAY_TASK_ID
python3 eval.py adam_mrhomo hmc_30x30x100_mis0 1000 $SLURM_ARRAY_TASK_ID

python3 eval.py adam_advhomo hmc_30x30x100_mis10 1000 $SLURM_ARRAY_TASK_ID
python3 eval.py adam_mrhomo hmc_30x30x100_mis10 1000 $SLURM_ARRAY_TASK_ID

python3 eval.py adam_advhomo hmc_30x30x100_mis20 1000 $SLURM_ARRAY_TASK_ID
python3 eval.py adam_mrhomo hmc_30x30x100_mis20 1000 $SLURM_ARRAY_TASK_ID


date
