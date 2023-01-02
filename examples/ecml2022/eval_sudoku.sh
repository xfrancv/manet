for ((task=0; task <= 175; task++))
do
   python3 ../../eval.py adam_mrhomo sudoku_1000_100_100_mis0 50 $task
   python3 ../../eval.py adam_advhomo sudoku_1000_100_100_mis0 50 $task
   python3 ../../eval.py adam_mrhomo sudoku_1000_100_100_mis10 50 $task
   python3 ../../eval.py adam_advhomo sudoku_1000_100_100_mis10 50 $task
   python3 ../../eval.py adam_mrhomo sudoku_1000_100_100_mis20 50 $task
   python3 ../../eval.py adam_advhomo sudoku_1000_100_100_mis20 50 $task
done
