#!/bin/bash

for ((task=0; task <= 14; task++))
do
   python3 ../../train.py adam_mrhomo hmc_30x30x100_mis0 $task
   python3 ../../train.py adam_advhomo hmc_30x30x100_mis0 $task
   python3 ../../train.py adam_mrhomo hmc_30x30x100_mis10 $task
   python3 ../../train.py adam_advhomo hmc_30x30x100_mis10 $task
   python3 ../../train.py adam_mrhomo hmc_30x30x100_mis20 $task
   python3 ../../train.py adam_advhomo hmc_30x30x100_mis20 $task
done
