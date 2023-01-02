#!/bin/bash

for ((task=0; task <= 14; task++))
do
   python3 ../../train.py adam_advhomo sudoku_1000_100_100_mis20_2000 $task
   python3 ../../train.py adam_mrhomo sudoku_1000_100_100_mis20_2000 $task
   python3 ../../train.py adam_mrhomo sudoku_1000_100_100_mis10_2000 $task
   python3 ../../train.py adam_advhomo sudoku_1000_100_100_mis10_2000 $task
   python3 ../../train.py adam_mrhomo sudoku_1000_100_100_mis0_2000 $task
   python3 ../../train.py adam_advhomo sudoku_1000_100_100_mis0_2000 $task
done
