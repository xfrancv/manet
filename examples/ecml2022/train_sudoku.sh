#!/bin/bash

for ((task=0; task <= 14; task++))
do
   python3 ../../train.py adam_advhomo sudoku_1000_100_100_mis20 $task
   python3 ../../train.py adam_mrhomo sudoku_1000_100_100_mis20 $task
   python3 ../../train.py adam_mrhomo sudoku_1000_100_100_mis10 $task
   python3 ../../train.py adam_advhomo sudoku_1000_100_100_mis10 $task
   python3 ../../train.py adam_mrhomo sudoku_1000_100_100_mis0 $task
   python3 ../../train.py adam_advhomo sudoku_1000_100_100_mis0 $task
done
