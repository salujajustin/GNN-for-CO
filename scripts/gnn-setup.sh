#!/bin/bash
# @author: Justin Saluja
# @description: Deep Policy Dynamic Programming script inspired by Kool's dpdp README, adapted for AWS

# Exit when any command fails
set -e

# Clone repository 
git clone https://github.com/salujajustin/GNN-for-CO.git && cd deep-policy-DP/

## INSTALL ENVIRONMENT ##

# Inform bash of conda
source ~/anaconda3/etc/profile.d/conda.sh

# Create environment named 'dpdp' and activate
conda create -n dpdp python=3.8 scipy anaconda -y
conda activate dpdp

# Install some basic packages
conda install tqdm -y
pip install gdown

# Install PyTorch, see https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch -y

# If you want to train models, you need these packages
pip install tensorboardx==1.5 fastprogress==0.1.18

# We use cupy for some efficient operations
pip install cupy-cuda110

# Also some efficient sparse operations using https://github.com/rusty1s/pytorch_scatter
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html


## DOWNLOAD INSTANCES ##

mkdir -p data/
cd data/

# Download tsp datasets (22mb each)
gdown --id 1tlcHok1JhOtQZOIshoGtyM5P9dZfnYbZ # tsp100_validation_seed4321.pkl
gdown --id 1woyNI8CoDJ8hyFko4NBJ6HdF4UQA0S77 # tsp100_test_seed1234.pkl

cd ../


## DOWNLOAD PRETRAINED MODEL ##

# Download TSP pretrained models from https://github.com/chaitjo/graph-convnet-tsp
mkdir -p logs
cd logs
gdown --id 1qmk1_5a8XT_hrOV_i3uHM9tMVnZBFEAF
tar -xvzf old-tsp-models.tar.gz
mv tsp-models/* .
rm -rf tsp-models
rm old-tsp-models.tar.gz
cd ..


## DOWNLOAD HEATMAPS ##

mkdir -p results/
cd results/

# Download TSP heatmaps (400mb each)
mkdir -p tsp100_validation_seed4321/heatmaps && gdown --id 14sc6E1OdOBB8ZuCpaWltdBpdD8-g8XYK --output tsp100_validation_seed4321/heatmaps/heatmaps_tsp100.pkl
mkdir -p tsp100_test_seed1234/heatmaps && gdown --id 1fSU39SzUoNlSUJo7qOqe7eL45Wak00vH --output tsp100_test_seed1234/heatmaps/heatmaps_tsp100.pkl

cd ../..


