import os
import time
import torch
import itertools
import argparse
import numpy as np

from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from utils.dp_utils import evaluate_dp
from utils.parser_utils import parse_args
from utils.data_utils import unpack_heatmaps, pack_heatmaps
from utils.data_utils import save_dataset, load_heatmaps, outfile_name
from utils.functions import move_to, get_durations, compute_batch_costs

from utils.datasets import HeatmapDataset, TSP


def eval_tsp(data_path, beam, args):

    problem = TSP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = problem.make_dataset(filename=data_path, num_samples=args.val_size, offset=args.offset)   
    dataset = pack_heatmaps(dataset, args)

    results = _eval_tsp(problem, dataset, beam, args, device) 
    costs, tours = zip(*results)

    #  Save results
    outfile = outfile_name(data_path, beam, args)
    save_dataset((results, args), outfile)

    return costs, tours



def _eval_tsp(problem, dataset, beam, args, device):

    dataloader = DataLoader(dataset, batch_size=1)
    results = []

    for batch in tqdm(dataloader, disable=False):
        batch = move_to(batch, device)
        batch, heatmaps = unpack_heatmaps(batch)

        with torch.no_grad():
            sequences, costs, batch_size = evaluate_dp(
                    is_vrp=False, batch=batch, heatmaps=heatmaps, beam_size=beam, collapse=args.decode_strategy, 
                    score_function=args.score_function,heatmap_threshold=args.heatmap_threshold, knn=args.knn, verbose=args.verbose
                    )
            costs = compute_batch_costs(problem, batch, sequences, device=device, check_costs=costs)

        for seq, cost in zip(sequences, costs):
            seq = seq if seq is None else seq.tolist() 
            results.append((cost, seq))

    return results


if __name__ == "__main__":

    #  Gather command line arguments/default values
    args = parse_args()

    #  If testing more then one beam size or dataset, iterate over them
    for beam in args.beam_size:
        for data_path in args.datasets:
            eval_tsp(data_path, beam, args) 
