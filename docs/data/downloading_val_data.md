# Downloading Validation Data 

Run the following to download the validation instances (graphs) for testing models and solutions. 

```bash
pip install gdown

mkdir -p data/tsp
cd data/tsp

# Download tsp datasets (22mb each)
gdown --id 1tlcHok1JhOtQZOIshoGtyM5P9dZfnYbZ # tsp100_validation_seed4321.pkl
gdown --id 1woyNI8CoDJ8hyFko4NBJ6HdF4UQA0S77 # tsp100_test_seed1234.pkl

cd ../..

```
