for ((task=0; task <= 165; task++))
do
   python3 ../../eval.py adam_advhomo hmc_30x30x100_mis0 1000 $task
   python3 ../../eval.py adam_mrhomo hmc_30x30x100_mis0 1000 $task
   python3 ../../eval.py adam_advhomo hmc_30x30x100_mis10 1000 $task
   python3 ../../eval.py adam_mrhomo hmc_30x30x100_mis10 1000 $task
   python3 ../../eval.py adam_advhomo hmc_30x30x100_mis20 1000 $task
   python3 ../../eval.py adam_mrhomo hmc_30x30x100_mis20 1000 $task
done
