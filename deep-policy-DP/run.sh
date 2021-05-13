#!/bin/bash
# @author: Justin Saluja
# @description: Deep Policy Dynamic Programming testing command script 

# Exit when any command fails
set -e

# Evaluate dpdp
PROBLEM=tsp
INSTANCES=data/tsp100_validation_seed4321.pkl
DECODE=dpdp
SCORE=heatmap_potential
BEAMSIZE=1000
THRESHOLD=1e-5
HEATMAP=results/tsp100_validation_seed4321/heatmaps/heatmaps_tsp100.pkl

python eval.py $INSTANCES --problem $PROBLEM --decode_strategy $DECODE --score_function $SCORE --beam_size $BEAMSIZE --heatmap_threshold $THRESHOLD --heatmap $HEATMAP 


# Plot visualize results
# PROBLEM=tsp
# INSTANCES=data/tsp100_validation_seed4321.pkl
# # SOLUTIONS=results/tsp100_validation_seed4321/tsp100_validation_seed4321offs0n10000-heatmaps_tsp100-dpdp100000heatmap_potential-th1e-05.pkl
# SOLUTIONS=/home/justin/Documents/18-786/aws/results/tsp/tsp100_test_seed1234/main_results/beam100000.pkl
# HEATMAPS=results/tsp100_validation_seed4321/heatmaps/heatmaps_tsp100.pkl

# # python visualize.py --problem $PROBLEM --instances $INSTANCES --solutions $SOLUTIONS --heatmaps $HEATMAPS
# python visualize.py --problem $PROBLEM --instances $INSTANCES --solutions $SOLUTIONS
