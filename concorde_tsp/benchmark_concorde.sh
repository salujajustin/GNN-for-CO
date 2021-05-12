#!/bin/bash

# This script benchmarks the Concorde TSP solver and saves the tours

# spin progress bar, source: https://stackoverflow.com/questions/12498304/using-bash-to-display-a-progress-indicator
sp="/-\|"
sc=0
spin() {
   printf "\b${sp:sc++:1}"
   ((sc==${#sp})) && sc=0
}
endspin() {
   printf "\r%s\n" "$@"
}

TIMEFORMAT=%R
i=0
for file in *test*.tsp;
doy
  # time Concorde
  (time ~/concorde/concorde/TSP/./concorde "concorde_data/tsp100_test_seed1234_${i}.tsp") 2>> times.txt
  # save log 
  ~/concorde/concorde/TSP/./concorde "concorde_data/tsp100_test_seed1234_${i}.tsp" > concorde_data/"${i}_test.log"
  spin
done
endspin