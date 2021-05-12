# GNN-for-CO

This repositorty contains code for the testing, verification and reimplementation for [Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475), [An Efficient Graph Convolutional Network Technique for the Travelling Salesman Problem](https://arxiv.org/abs/1906.01227), and [Deep Policy Dynamic Programming](https://arxiv.org/abs/2102.11756)

![pipeline](res/tsp.png)

### Concorde TSP Solver

The Concorde TSP Solver can be installed using the following script referenced from [attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/tsp/install_concorde.sh):

```bash
bash concorde_tsp/install_concorde.sh
```
In accordance with the accompanying wrappers, Concorde should be placed at the root directory. The validation data on which Concorde will run must be fetched prior to running the script by following the instructions in ```docs/data/downloading_val_data.md```. These data must be placed within the ```concorde_tsp``` directory.

The following Python dependencies must be installed:
- pickle
- sys
- os
- numpy
- tqdm 
- matplotlib.pyplot

Then, the Concorde TSP benchmarking and plotting routine can be run in Python 3.8.5 as follows:
```
mkdir concorde_data
python concorde_tsp/concorde_format_data.py
```
This wrapper will first generate ```.tsp``` files in the directory ```concorde_data```. Then, Concorde will be run on all data to output ```.log``` files containing the runtime history and ```.sol``` files containing the optimal tours. Finally, the tours will be consolidated and summarized, and a plot will be generated.

### Greedy Search

The greedy search algorithm benchmarking and plotting routine can be run in Python 3.8.5 as follows:
```bash
python greedy_search/greedy_search.py
```

The validation data on which greedy search will run must be fetched prior to running the script by following the instructions in ```docs/data/downloading_val_data.md```.

The following Python dependencies must be installed:
- pickle
- sys
- os
- numpy
- tqdm
- matplotlib
- time

### Deep Policy Dynamic Programming

Run the following to setup a the environment on AWS Deep Learning AMI (Ubuntu 18.04) Version 43.0 : 
```bash

# Install Python 3.8
sudo apt install python3.8

# Download environment setup script
wget https://raw.githubusercontent.com/salujajustin/GNN-for-CO/main/scripts/dpdp-setup.sh

# Change permissions and run
chmod +x dpdp-setup.sh
./dpdp-setup.sh

# Activate environment
conda activate dpdp
```


## Acknowledgements
This repository was built upon the excellent repositories of [graph-convnet-tsp](https://github.com/chaitjo/graph-convnet-tsp) by Chaitanya K. Joshi and [dpdp](https://github.com/wouterkool/dpdp) by Wouter Kool.
