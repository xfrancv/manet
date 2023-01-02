for ((task=0; task <= 175; task++))
do
   python3 ../../eval.py adam_mrhomo sudoku_1000_100_100_mis0_2000 50 $task
   python3 ../../eval.py adam_advhomo sudoku_1000_100_100_mis0_2000 50 $task
   python3 ../../eval.py adam_mrhomo sudoku_1000_100_100_mis10_2000 50 $task
   python3 ../../eval.py adam_advhomo sudoku_1000_100_100_mis10_2000 50 $task
   python3 ../../eval.py adam_mrhomo sudoku_1000_100_100_mis20_2000 50 $task
   python3 ../../eval.py adam_advhomo sudoku_1000_100_100_mis20_2000 50 $task
done
